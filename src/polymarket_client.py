"""Polymarket CLOB API client for 5-min BTC Up/Down market trading.

Handles:
- Authentication via private key + derived API credentials
- Market discovery via Gamma API for the next 5-min BTC slot
- Order placement (market buy) on the correct Up/Down token
- Balance fetching, open positions, connection health checks
- Duplicate trade prevention per slot

Requires:
    py-clob-client >= 0.34.6
    Environment variables: POLYMARKET_PRIVATE_KEY, POLYMARKET_FUNDER_ADDRESS,
                           POLYMARKET_SIGNATURE_TYPE (optional, default 2)
"""
import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# Polymarket API endpoints
CLOB_HOST = "https://clob.polymarket.com"
GAMMA_API = "https://gamma-api.polymarket.com"
DATA_API = "https://data-api.polymarket.com"

# Polygon mainnet chain ID
CHAIN_ID = 137

# 5-minute candle period in seconds
SLOT_PERIOD = 300


class PolymarketClient:
    """Client for Polymarket CLOB API, specialized for 5-min BTC Up/Down markets."""

    def __init__(self, private_key: str, funder_address: str, signature_type: int = 2):
        """
        Args:
            private_key: Wallet private key (hex, with or without 0x prefix)
            funder_address: Funder/proxy wallet address
            signature_type: Signature type (0, 1, or 2). Default: 2
        """
        self._private_key = private_key
        self._funder_address = funder_address
        self._signature_type = signature_type
        self._client = None  # ClobClient instance (lazy init)
        self._api_creds = None
        self._initialized = False
        self._last_traded_slot: Optional[int] = None  # Unix ts of last traded slot
        self._http = httpx.AsyncClient(timeout=15)

    async def initialize(self) -> dict:
        """Initialize the CLOB client and derive API credentials.

        Returns:
            {success: bool, error: str|None}
        """
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import ApiCreds

            self._client = ClobClient(
                host=CLOB_HOST,
                key=self._private_key,
                chain_id=CHAIN_ID,
                signature_type=self._signature_type,
                funder=self._funder_address,
            )

            # Derive or retrieve API credentials
            self._api_creds = self._client.create_or_derive_api_creds()
            self._client.set_api_creds(self._api_creds)

            self._initialized = True
            logger.info("Polymarket client initialized successfully")
            return {"success": True, "error": None}

        except Exception as e:
            logger.error(f"Polymarket client initialization failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def wallet_address(self) -> str:
        """Return the funder address (display purposes)."""
        return self._funder_address

    # ------------------------------------------------------------------
    # Balance
    # ------------------------------------------------------------------

    async def get_balance(self) -> dict:
        """Fetch USDC balance from Polymarket.

        Returns:
            {success: bool, data: {balance: float, currency: str}, error: str|None}
        """
        if not self._initialized:
            return {"success": False, "data": None, "error": "Client not initialized"}

        try:
            balance = self._client.get_balance()
            bal_float = float(balance) if balance is not None else 0.0
            logger.info(f"Polymarket balance: {bal_float:.2f} USDC")
            return {
                "success": True,
                "data": {"balance": bal_float, "currency": "USDC"},
                "error": None,
            }
        except Exception as e:
            logger.error(f"Failed to fetch balance: {e}", exc_info=True)
            return {"success": False, "data": None, "error": str(e)}

    # ------------------------------------------------------------------
    # Market Discovery
    # ------------------------------------------------------------------

    @staticmethod
    def get_next_slot_timestamp() -> int:
        """Compute the Unix timestamp of the next 5-min slot opening.

        E.g. if now is 09:03:22 -> next slot opens at 09:05:00 -> returns 09:05:00 as Unix ts.
        """
        now_ts = int(time.time())
        return now_ts - (now_ts % SLOT_PERIOD) + SLOT_PERIOD

    @staticmethod
    def slot_to_datetime(slot_ts: int) -> datetime:
        """Convert a Unix slot timestamp to a UTC datetime."""
        return datetime.fromtimestamp(slot_ts, tz=timezone.utc)

    async def get_current_market(self) -> dict:
        """Find the active Polymarket 5-min BTC Up/Down market for the next slot.

        Searches Gamma API for the upcoming BTC 5-min binary market.

        Returns:
            {success: bool, data: {condition_id, up_token_id, down_token_id,
             slot_ts, slot_dt, question, outcomes, prices}, error: str|None}
        """
        if not self._initialized:
            return {"success": False, "data": None, "error": "Client not initialized"}

        try:
            # Search for active BTC 5-min up/down markets
            resp = await self._http.get(
                f"{GAMMA_API}/markets",
                params={
                    "closed": "false",
                    "active": "true",
                    "limit": 20,
                    "order": "volume_24hr",
                    "ascending": "false",
                },
            )
            resp.raise_for_status()
            markets = resp.json()

            # Find the BTC 5-min up/down market
            target_market = None
            for market in markets:
                question = (market.get("question") or "").lower()
                # Match patterns like "BTC 5 min", "Bitcoin 5-minute", "BTC up or down"
                if ("btc" in question or "bitcoin" in question) and \
                   ("5 min" in question or "5-min" in question or "5min" in question) and \
                   ("up" in question and "down" in question):
                    target_market = market
                    break

            # Fallback: search via public-search endpoint
            if target_market is None:
                resp2 = await self._http.get(
                    f"{GAMMA_API}/public-search",
                    params={"query": "BTC 5 min up down", "limit": 10},
                )
                resp2.raise_for_status()
                search_results = resp2.json()

                # Search results can be events or markets
                for item in search_results:
                    # If it's an event with nested markets
                    if "markets" in item:
                        for m in item["markets"]:
                            q = (m.get("question") or "").lower()
                            if ("btc" in q or "bitcoin" in q) and \
                               ("5 min" in q or "5-min" in q or "5min" in q) and \
                               ("up" in q and "down" in q) and \
                               not m.get("closed", True):
                                target_market = m
                                break
                    else:
                        q = (item.get("question") or "").lower()
                        if ("btc" in q or "bitcoin" in q) and \
                           ("5 min" in q or "5-min" in q or "5min" in q) and \
                           ("up" in q and "down" in q) and \
                           not item.get("closed", True):
                            target_market = item
                            break
                    if target_market:
                        break

            if target_market is None:
                return {
                    "success": False,
                    "data": None,
                    "error": "BTC 5-min Up/Down market not found on Polymarket",
                }

            # Parse market data
            condition_id = target_market.get("conditionId", "")

            # clobTokenIds is a JSON string array: '["id1", "id2"]'
            clob_token_ids_raw = target_market.get("clobTokenIds", "[]")
            if isinstance(clob_token_ids_raw, str):
                clob_token_ids = json.loads(clob_token_ids_raw)
            else:
                clob_token_ids = clob_token_ids_raw

            # outcomes is a JSON string array: '["Up", "Down"]'
            outcomes_raw = target_market.get("outcomes", "[]")
            if isinstance(outcomes_raw, str):
                outcomes = json.loads(outcomes_raw)
            else:
                outcomes = outcomes_raw

            # outcomePrices is a JSON string array: '["0.50", "0.50"]'
            prices_raw = target_market.get("outcomePrices", "[]")
            if isinstance(prices_raw, str):
                prices = json.loads(prices_raw)
            else:
                prices = prices_raw

            # Map outcomes to token IDs
            # outcomes[i] corresponds to clobTokenIds[i]
            up_token_id = None
            down_token_id = None
            for i, outcome in enumerate(outcomes):
                outcome_lower = outcome.lower().strip()
                if outcome_lower in ("up", "yes") and i < len(clob_token_ids):
                    up_token_id = clob_token_ids[i]
                elif outcome_lower in ("down", "no") and i < len(clob_token_ids):
                    down_token_id = clob_token_ids[i]

            # Fallback: if outcomes are just ["Yes", "No"] or similar
            if up_token_id is None and len(clob_token_ids) >= 1:
                up_token_id = clob_token_ids[0]
            if down_token_id is None and len(clob_token_ids) >= 2:
                down_token_id = clob_token_ids[1]

            next_slot_ts = self.get_next_slot_timestamp()
            slot_dt = self.slot_to_datetime(next_slot_ts)

            market_data = {
                "condition_id": condition_id,
                "up_token_id": up_token_id,
                "down_token_id": down_token_id,
                "slot_ts": next_slot_ts,
                "slot_dt": slot_dt.isoformat(),
                "question": target_market.get("question", "N/A"),
                "outcomes": outcomes,
                "prices": [float(p) for p in prices] if prices else [],
                "market_slug": target_market.get("slug", ""),
                "enable_order_book": target_market.get("enableOrderBook", False),
            }

            logger.info(
                f"Market found: {market_data['question']} | "
                f"Up={up_token_id[:12]}... Down={down_token_id[:12]}..."
            )
            return {"success": True, "data": market_data, "error": None}

        except httpx.HTTPStatusError as e:
            logger.error(f"Gamma API HTTP error: {e.response.status_code}")
            return {"success": False, "data": None, "error": f"Gamma API error: {e.response.status_code}"}
        except Exception as e:
            logger.error(f"Market discovery failed: {e}", exc_info=True)
            return {"success": False, "data": None, "error": str(e)}

    # ------------------------------------------------------------------
    # Order Book
    # ------------------------------------------------------------------

    async def get_best_price(self, token_id: str, side: str = "BUY") -> Optional[float]:
        """Get the best available price from the order book.

        Args:
            token_id: The CLOB token ID
            side: "BUY" or "SELL"

        Returns:
            Best price as float, or None if no orders available.
        """
        try:
            resp = await self._http.get(
                f"{CLOB_HOST}/order-book/{token_id}"
            )
            resp.raise_for_status()
            book = resp.json()

            if side == "BUY":
                # Best ask price (lowest sell order) for buying
                asks = book.get("asks", [])
                if asks:
                    return float(asks[0]["price"])
            else:
                # Best bid price (highest buy order) for selling
                bids = book.get("bids", [])
                if bids:
                    return float(bids[0]["price"])

            return None
        except Exception as e:
            logger.error(f"Order book fetch failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Trade Execution
    # ------------------------------------------------------------------

    async def place_trade(self, direction: str, amount: float) -> dict:
        """Place a market buy order on the correct Up/Down token.

        Args:
            direction: "UP" or "DOWN"
            amount: Trade size in USDC

        Returns:
            {success: bool, data: {order_id, direction, amount, price,
             token_id, slot_ts, slot_dt, question, filled_at}, error: str|None}
        """
        if not self._initialized:
            return {"success": False, "data": None, "error": "Client not initialized"}

        if direction not in ("UP", "DOWN"):
            return {"success": False, "data": None, "error": f"Invalid direction: {direction}"}

        try:
            # Step 1: Discover the current market
            market_result = await self.get_current_market()
            if not market_result["success"]:
                return {
                    "success": False,
                    "data": None,
                    "error": f"Market discovery failed: {market_result['error']}",
                }

            market = market_result["data"]
            slot_ts = market["slot_ts"]

            # Step 2: Duplicate trade prevention
            if self._last_traded_slot == slot_ts:
                return {
                    "success": False,
                    "data": None,
                    "error": f"Duplicate trade prevented: already traded slot {market['slot_dt']}",
                }

            # Step 3: Pick the correct token
            if direction == "UP":
                token_id = market["up_token_id"]
            else:
                token_id = market["down_token_id"]

            if not token_id:
                return {
                    "success": False,
                    "data": None,
                    "error": f"No token ID found for {direction}",
                }

            # Step 4: Get best available price from order book
            best_price = await self.get_best_price(token_id, side="BUY")
            if best_price is None:
                # Use 0.50 as a reasonable default for binary markets
                best_price = 0.50
                logger.warning(f"No asks in order book, using default price {best_price}")

            # Step 5: Calculate size (number of outcome tokens)
            # size = USDC amount / price per token
            size = round(amount / best_price, 2)

            # Step 6: Place the order
            logger.info(
                f"Placing {direction} order: token={token_id[:16]}..., "
                f"price={best_price}, size={size}, amount={amount} USDC"
            )

            order_resp = self._client.create_and_post_order(
                token_id=token_id,
                price=best_price,
                size=size,
                side="BUY",
            )

            # Mark slot as traded
            self._last_traded_slot = slot_ts

            # Parse order response
            order_id = "unknown"
            if isinstance(order_resp, dict):
                order_id = order_resp.get("orderID", order_resp.get("id", "unknown"))
            elif hasattr(order_resp, "orderID"):
                order_id = order_resp.orderID
            elif hasattr(order_resp, "id"):
                order_id = order_resp.id

            trade_data = {
                "order_id": str(order_id),
                "direction": direction,
                "amount": amount,
                "price": best_price,
                "size": size,
                "token_id": token_id,
                "slot_ts": slot_ts,
                "slot_dt": market["slot_dt"],
                "question": market["question"],
                "filled_at": datetime.now(timezone.utc).isoformat(),
            }

            logger.info(
                f"Trade placed: {direction} | order={order_id} | "
                f"price={best_price} | size={size} | slot={market['slot_dt']}"
            )
            return {"success": True, "data": trade_data, "error": None}

        except Exception as e:
            logger.error(f"Trade execution failed: {e}", exc_info=True)
            return {"success": False, "data": None, "error": str(e)}

    # ------------------------------------------------------------------
    # Open Positions
    # ------------------------------------------------------------------

    async def get_open_positions(self) -> dict:
        """Fetch open positions from the Polymarket Data API.

        Returns:
            {success: bool, data: list[dict], error: str|None}
        """
        if not self._initialized:
            return {"success": False, "data": None, "error": "Client not initialized"}

        try:
            resp = await self._http.get(
                f"{DATA_API}/positions",
                params={"user": self._funder_address},
            )
            resp.raise_for_status()
            positions = resp.json()

            # Normalize positions data
            formatted = []
            if isinstance(positions, list):
                for pos in positions:
                    formatted.append({
                        "market": pos.get("title", pos.get("question", "Unknown")),
                        "outcome": pos.get("outcome", "N/A"),
                        "size": float(pos.get("size", 0)),
                        "avg_price": float(pos.get("avgPrice", pos.get("price", 0))),
                        "current_value": float(pos.get("currentValue", pos.get("value", 0))),
                        "pnl": float(pos.get("pnl", pos.get("realizedPnl", 0))),
                        "token_id": pos.get("asset", pos.get("tokenId", "")),
                    })

            logger.info(f"Fetched {len(formatted)} open positions")
            return {"success": True, "data": formatted, "error": None}

        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}", exc_info=True)
            return {"success": False, "data": None, "error": str(e)}

    # ------------------------------------------------------------------
    # Health Check
    # ------------------------------------------------------------------

    async def is_connected(self) -> dict:
        """Check Polymarket connection health.

        Returns:
            {connected: bool, balance: float|None, error: str|None}
        """
        if not self._initialized:
            return {"connected": False, "balance": None, "error": "Client not initialized"}

        try:
            # Check CLOB API health
            resp = await self._http.get(f"{CLOB_HOST}/")
            api_ok = resp.status_code == 200

            # Check balance
            bal_result = await self.get_balance()
            balance = bal_result["data"]["balance"] if bal_result["success"] else None

            connected = api_ok and bal_result["success"]
            return {
                "connected": connected,
                "balance": balance,
                "error": None if connected else "API or balance check failed",
            }

        except Exception as e:
            return {"connected": False, "balance": None, "error": str(e)}

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def close(self):
        """Close HTTP client."""
        await self._http.aclose()
