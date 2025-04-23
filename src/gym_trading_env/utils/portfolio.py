from typing import Dict, List, Tuple, Union, Optional
import numpy as np

class Portfolio:
    """
    Represents a trading portfolio with balances for one base instrument
    and one quote currency, including borrowed amounts and accrued interest.
    Suitable for single-instrument trading environments (e.g., one stock vs USD).
    """
    def __init__(self, base: float, quote: float, interest_base: float = 0, interest_quote: float = 0):
        """
        Initializes the Portfolio.

        Args:
            base: Initial amount of the base instrument (e.g., number of shares, amount of crypto).
            quote: Initial amount of the quote currency (e.g., USD, EUR).
            interest_base: Initial accrued interest on borrowed base instrument. Defaults to 0.
            interest_quote: Initial accrued interest on borrowed quote currency. Defaults to 0.
        """
        self.base = base
        self.quote = quote
        self.interest_base = interest_base
        self.interest_quote = interest_quote

    def valuation(self, price: float) -> float:
        """
        Calculates the total value of the portfolio in the quote currency at a given price,
        considering base instrument, quote currency, and subtracting accrued interest.

        Args:
            price: The current price of the base instrument in the quote currency.

        Returns:
            The total portfolio value in the quote currency.
        """
        return sum([
            self.base * price,
            self.quote,
            - self.interest_base * price, # Subtract interest owed in base value (converted to quote)
            - self.interest_quote         # Subtract interest owed in quote value
        ])

    def real_position(self, price: float) -> float:
        """
        Calculates the 'real' position ratio, representing the net base instrument exposure
        (base minus borrowed base interest, converted to quote currency) as a fraction
        of the total portfolio value.

        Args:
            price: The current price of the base instrument.

        Returns:
            The real position ratio. Returns 0.0 if valuation is zero.
        """
        portfolio_value = self.valuation(price)
        if portfolio_value == 0:
            return 0.0 # Avoid division by zero; position is effectively zero
        return (self.base - self.interest_base) * price / portfolio_value

    def position(self, price: float) -> float:
        """
        Calculates the 'gross' position ratio, representing the total base instrument value
        (converted to quote currency) as a fraction of the total portfolio value.

        Args:
            price: The current price of the base instrument.

        Returns:
            The gross position ratio. Returns 0.0 if valuation is zero.
        """
        portfolio_value = self.valuation(price)
        if portfolio_value == 0:
            return 0.0 # Avoid division by zero; position is effectively zero
        return self.base * price / portfolio_value

    def trade_to_position(self, position: float, price: float, trading_fees: float):
        """
        Adjusts the portfolio holdings to reach a target position ratio.
        Handles interest repayment proportionally when reducing leverage,
        and accounts for trading fees. Assumes a single base instrument vs quote currency.

        Args:
            position: The target position ratio (0 for full quote, 1 for full base).
            price: The current price of the base instrument for trading.
            trading_fees: The fee rate applied to trades (e.g., 0.001 for 0.1%).
        """
        current_valuation = self.valuation(price)
        if current_valuation <= 0: # Cannot trade if portfolio has no value or is negative
             return

        # --- Step 1: Repay Interest Proportionally (if reducing leverage) ---
        # Calculate how much of the existing interest should be kept based on the new position.
        # If moving towards a less leveraged position (closer to [0, 1]),
        # repay a portion of the outstanding interest.
        current_position = self.position(price)
        interest_reduction_ratio = 1.0 # Default: keep all interest

        # If currently short base (pos < 0) and moving towards less short
        if position <= 0 and current_position < 0:
            # Denominator current_position cannot be 0 here
            interest_reduction_ratio = min(1.0, max(0.0, position / current_position))
        # If currently long leveraged base (pos > 1) and moving towards less leverage
        elif position >= 1 and current_position > 1:
            # Denominator (current_position - 1) cannot be 0 here
            interest_reduction_ratio = min(1.0, max(0.0, (position - 1) / (current_position - 1)))

        if interest_reduction_ratio < 1:
            repayment_ratio = 1 - interest_reduction_ratio
            # Reduce base/quote by the amount of interest being repaid
            self.base -= repayment_ratio * self.interest_base
            self.quote -= repayment_ratio * self.interest_quote
            # Reduce the outstanding interest accordingly
            self.interest_base *= interest_reduction_ratio
            self.interest_quote *= interest_reduction_ratio
            # Recalculate valuation after interest repayment
            current_valuation = self.valuation(price)
            if current_valuation <= 0: return # Stop if repayment made value non-positive


        # --- Step 2: Calculate and Execute Trade ---
        # Target base instrument value based on the desired position and current valuation
        target_base_value = position * current_valuation
        target_base_amount = target_base_value / price if price != 0 else 0

        # Amount of base instrument to trade (positive: buy base, negative: sell base)
        base_trade = target_base_amount - self.base

        if base_trade == 0: # No trade needed
            return

        # Adjust trade amount for fees
        # Fee on BUY base: Deducted from the base received.
        # Fee on SELL base: Deducted from the quote received.

        if base_trade > 0: # BUYING base instrument
            # Amount of base we need to *receive* after fees
            net_base_increase_needed = target_base_amount - self.base
            # Gross amount of base to buy before fees
            gross_base_to_buy = net_base_increase_needed / (1 - trading_fees) if (1 - trading_fees) != 0 else net_base_increase_needed
            # Quote currency cost of the gross buy
            quote_cost = -gross_base_to_buy * price

            self.base += gross_base_to_buy * (1 - trading_fees) # Receive net amount
            self.quote += quote_cost

        else: # SELLING base instrument (base_trade is negative)
            # Amount of base we need to sell
            base_to_sell = -base_trade # This is a positive number
            # Quote received before fees
            gross_quote_received = base_to_sell * price
            # Fee amount (paid from quote received)
            fee = gross_quote_received * trading_fees
            # Net quote received after fees
            net_quote_received = gross_quote_received * (1 - trading_fees)

            self.base += base_trade # Decrease base amount (base_trade is negative)
            self.quote += net_quote_received

    def update_interest(self, borrow_interest_rate: float):
        """
        Updates the accrued interest based on borrowed amounts (negative balances).
        Interest is calculated per step (needs rate adjusted accordingly).

        Args:
            borrow_interest_rate: The interest rate per step for borrowing (applied to both base and quote).
        """
        # Interest accrues only on borrowed amounts (negative balances)
        self.interest_base += max(0, -self.base) * borrow_interest_rate
        self.interest_quote += max(0, -self.quote) * borrow_interest_rate

    def __str__(self) -> str:
        """Returns a string representation of the portfolio."""
        # Use f-string interpolation correctly
        return f"{self.__class__.__name__}(base={self.base}, quote={self.quote}, interest_base={self.interest_base}, interest_quote={self.interest_quote})"


    def describe(self, price: float):
        """Prints the current portfolio value (in quote currency) and position."""
        print(f"Value : {self.valuation(price):.2f}, Position : {self.position(price):.3f}")

    def get_portfolio_distribution(self) -> dict:
        """
        Returns a dictionary describing the distribution of the portfolio components.

        Returns:
            A dictionary containing base, quote, borrowed amounts, and interest.
        """
        return {
            "base": max(0.0, self.base),
            "quote": max(0.0, self.quote),
            "borrowed_base": max(0.0, -self.base),
            "borrowed_quote": max(0.0, -self.quote),
            "interest_base": self.interest_base,
            "interest_quote": self.interest_quote,
        }

    @classmethod
    def from_multi_asset(cls, multi_portfolio: 'MultiAssetPortfolio', base_symbol: str, price: float):
        """
        Creates a single-asset Portfolio from a MultiAssetPortfolio focused on one asset.

        Args:
            multi_portfolio: The source MultiAssetPortfolio
            base_symbol: The symbol of the asset to use as base
            price: The current price of the base asset

        Returns:
            A new Portfolio instance with the extracted base asset and quote currency
        """
        # Extract the base asset quantity and interest
        base = multi_portfolio.assets.get(base_symbol, 0.0)
        interest_base = multi_portfolio.interest_assets.get(base_symbol, 0.0)

        # Use the quote as is
        quote = multi_portfolio.quote
        interest_quote = multi_portfolio.interest_quote

        return cls(
            base=base,
            quote=quote,
            interest_base=interest_base,
            interest_quote=interest_quote
        )

    def to_multi_asset(self, base_symbol: str) -> 'MultiAssetPortfolio':
        """
        Converts this single-asset Portfolio to a MultiAssetPortfolio.

        Args:
            base_symbol: The symbol to use for the base asset

        Returns:
            A new MultiAssetPortfolio with this portfolio's assets
        """
        multi = MultiAssetPortfolio(
            assets={base_symbol: self.base} if self.base != 0 else {},
            quote=self.quote,
            interest_assets={base_symbol: self.interest_base} if self.interest_base != 0 else {},
            interest_quote=self.interest_quote
        )
        return multi


class MultiAssetPortfolio:
    """
    Represents a trading portfolio with multiple assets and a quote currency.
    Supports borrowed amounts and interest tracking for all components.
    """
    def __init__(
        self, 
        assets: Dict[str, float] = None,
        quote: float = 0.0,
        interest_assets: Dict[str, float] = None,
        interest_quote: float = 0.0
    ):
        """
        Initializes the MultiAssetPortfolio.

        Args:
            assets: Dictionary mapping asset symbols to quantities. Default empty dict.
            quote: Amount of the quote currency (e.g., USD, EUR). Default 0.
            interest_assets: Dictionary mapping asset symbols to interest amounts. Default empty dict.
            interest_quote: Accrued interest on borrowed quote currency. Default 0.
        """
        self.assets = assets or {}
        self.quote = quote
        self.interest_assets = interest_assets or {}
        self.interest_quote = interest_quote

        # Ensure all assets have interest entries (default 0)
        for asset in self.assets:
            if asset not in self.interest_assets:
                self.interest_assets[asset] = 0.0

    def valuation(self, prices: Dict[str, float]) -> float:
        """
        Calculates the total portfolio value in quote currency.

        Args:
            prices: Dictionary mapping asset symbols to their current prices

        Returns:
            The total portfolio value in quote currency
        """
        # Start with quote currency amount
        total_value = self.quote - self.interest_quote

        # Add value of each asset
        for symbol, amount in self.assets.items():
            if symbol in prices:
                # Add value of asset (or negative if borrowed)
                total_value += amount * prices[symbol]

                # Subtract interest on this asset
                interest = self.interest_assets.get(symbol, 0.0)
                total_value -= interest * prices[symbol]

        return total_value

    def asset_allocations(self, prices: Dict[str, float]) -> Dict[str, float]:
        """
        Calculates the allocation ratio of each asset in the portfolio.

        Args:
            prices: Dictionary mapping asset symbols to their current prices

        Returns:
            Dictionary mapping asset symbols to their allocation ratios (0-1)
        """
        total_value = self.valuation(prices)
        if total_value <= 0:
            return {symbol: 0.0 for symbol in self.assets}

        allocations = {}
        for symbol, amount in self.assets.items():
            if symbol in prices:
                allocations[symbol] = amount * prices[symbol] / total_value

        return allocations

    def real_asset_allocations(self, prices: Dict[str, float]) -> Dict[str, float]:
        """
        Calculates the real allocation ratio of each asset (net of interest).

        Args:
            prices: Dictionary mapping asset symbols to their current prices

        Returns:
            Dictionary mapping asset symbols to their real allocation ratios
        """
        total_value = self.valuation(prices)
        if total_value <= 0:
            return {symbol: 0.0 for symbol in self.assets}

        allocations = {}
        for symbol, amount in self.assets.items():
            if symbol in prices:
                net_amount = amount - self.interest_assets.get(symbol, 0.0)
                allocations[symbol] = net_amount * prices[symbol] / total_value

        return allocations

    def trade_asset(self, symbol: str, target_allocation: float, prices: Dict[str, float], trading_fees: float = 0.0) -> bool:
        """
        Trades a single asset to target a specific allocation in the portfolio.

        Args:
            symbol: The asset symbol to trade
            target_allocation: The target allocation ratio for this asset (0-1)
            prices: Dictionary of asset prices
            trading_fees: Fee rate applied to trades

        Returns:
            True if trade was successful, False otherwise
        """
        if symbol not in prices:
            return False

        price = prices[symbol]
        if price <= 0:
            return False

        # Calculate portfolio valuation
        current_value = self.valuation(prices)
        if current_value <= 0:
            return False

        # Current allocation
        current_amount = self.assets.get(symbol, 0.0)
        current_allocation = current_amount * price / current_value

        # Target amount of the asset
        target_value = target_allocation * current_value
        target_amount = target_value / price

        # Amount to trade (positive: buy, negative: sell)
        trade_amount = target_amount - current_amount

        if abs(trade_amount) < 1e-8:  # No significant trade needed
            return True

        # Handle interest reduction for this asset if relevant
        if self._handle_interest_reduction(symbol, target_allocation, current_allocation, price) is False:
            return False

        # Execute the trade with fees
        if trade_amount > 0:  # Buying
            # Amount needed after fees
            gross_amount = trade_amount / (1 - trading_fees)
            quote_cost = -gross_amount * price

            # Update balances
            self.assets[symbol] = self.assets.get(symbol, 0.0) + gross_amount * (1 - trading_fees)
            self.quote += quote_cost
        else:  # Selling
            amount_to_sell = -trade_amount
            gross_quote_received = amount_to_sell * price
            net_quote_received = gross_quote_received * (1 - trading_fees)

            # Update balances
            current_amount = self.assets.get(symbol, 0.0)
            new_amount = current_amount - amount_to_sell

            if new_amount == 0:
                del self.assets[symbol]  # Remove the asset if fully sold
            else:
                self.assets[symbol] = new_amount

            self.quote += net_quote_received

        return True

    def _handle_interest_reduction(self, symbol: str, target_allocation: float, current_allocation: float, price: float) -> bool:
        """
        Handles interest reduction when moving towards less leveraged positions.

        Args:
            symbol: The asset symbol
            target_allocation: Target allocation ratio
            current_allocation: Current allocation ratio
            price: Current price

        Returns:
            True if successful, False if fails (e.g., due to negative valuation)
        """
        # Only applicable if the asset has interest and is reducing leverage
        if symbol not in self.interest_assets or self.interest_assets[symbol] <= 0:
            return True

        interest_reduction_ratio = 1.0  # Default: keep all interest

        # If short position (negative allocation) moving toward less short
        if target_allocation <= 0 and current_allocation < 0:
             # Denominator current_allocation cannot be 0 here
            interest_reduction_ratio = min(1.0, target_allocation / current_allocation)
        # If leveraged long position moving toward less leverage
        elif target_allocation >= 1 and current_allocation > 1:
             # Denominator (current_allocation - 1) cannot be 0 here
            interest_reduction_ratio = min(1.0, (target_allocation - 1) / (current_allocation - 1))

        if interest_reduction_ratio < 1.0:
            repayment_ratio = 1.0 - interest_reduction_ratio

            # Reduce asset by interest being repaid
            repayment_amount = repayment_ratio * self.interest_assets[symbol]
            self.assets[symbol] -= repayment_amount

            # Reduce quote by equivalent value
            self.quote -= repayment_amount * price

            # Reduce outstanding interest
            self.interest_assets[symbol] *= interest_reduction_ratio

            # Check if valuation still positive
            if self.valuation({k: v for k, v in price.items() if k == symbol}) <= 0:
                return False

        return True

    def trade_to_allocations(self, target_allocations: Dict[str, float], prices: Dict[str, float], trading_fees: float = 0.0) -> bool:
        """
        Trades multiple assets to achieve target allocations.
        Handles cash balancing automatically.

        Args:
            target_allocations: Dict mapping symbols to target allocation ratios
            prices: Dict mapping symbols to current prices
            trading_fees: Fee rate for trading

        Returns:
            True if all trades were successful, False otherwise
        """
        # Validate allocations sum to approximately 1.0
        allocation_sum = sum(target_allocations.values())
        if not (0.99 <= allocation_sum <= 1.01):
            return False

        # Sort assets by decreasing distance from target allocation 
        current_allocations = self.asset_allocations(prices)

        assets_to_trade = []
        for symbol, target in target_allocations.items():
            current = current_allocations.get(symbol, 0.0)
            distance = abs(target - current)
            assets_to_trade.append((symbol, distance))

        # Add assets we own but aren't in target (to sell)
        for symbol in self.assets:
            if symbol not in target_allocations and symbol in prices:
                current = current_allocations.get(symbol, 0.0)
                assets_to_trade.append((symbol, current))

        # Sort by distance descending
        assets_to_trade.sort(key=lambda x: x[1], reverse=True)

        # Trade each asset
        success = True
        for symbol, _ in assets_to_trade:
            target = target_allocations.get(symbol, 0.0)
            if not self.trade_asset(symbol, target, prices, trading_fees):
                success = False

        return success

    def update_interest(self, borrow_interest_rate: float):
        """
        Updates accrued interest on borrowed assets and quote currency.

        Args:
            borrow_interest_rate: The interest rate per step for borrowing
        """
        # Update interest on quote currency if borrowed
        if self.quote < 0:
            self.interest_quote += (-self.quote) * borrow_interest_rate

        # Update interest on each borrowed asset
        for symbol, amount in list(self.assets.items()):
            if amount < 0:  # Borrowed
                self.interest_assets[symbol] = self.interest_assets.get(symbol, 0.0) + (-amount) * borrow_interest_rate

    def get_portfolio_distribution(self) -> dict:
        """
        Returns a dictionary describing the distribution of portfolio components.

        Returns:
            Dict with asset, quote, and interest information
        """
        result = {
            "quote": max(0.0, self.quote),
            "borrowed_quote": max(0.0, -self.quote),
            "interest_quote": self.interest_quote
        }

        # Add each asset's info
        for symbol, amount in self.assets.items():
            if amount > 0:
                result[f"asset_{symbol}"] = amount
            else:
                result[f"borrowed_asset_{symbol}"] = -amount

            interest = self.interest_assets.get(symbol, 0.0)
            if interest > 0:
                result[f"interest_asset_{symbol}"] = interest

        return result

    def __str__(self) -> str:
        """Returns a string representation of the portfolio."""
        assets_str = ", ".join([f"{s}:{a}" for s, a in self.assets.items()])
        return f"MultiAssetPortfolio(assets={{{assets_str}}}, quote={self.quote}, interest_assets={self.interest_assets}, interest_quote={self.interest_quote})"


class TargetMultiAssetPortfolio(MultiAssetPortfolio):
    """
    A MultiAssetPortfolio initialized with target allocations and a starting value.
    """
    def __init__(self, allocations: Dict[str, float], total_value: float, prices: Dict[str, float]):
        """
        Initializes the portfolio based on target allocations.

        Args:
            allocations: Dict mapping asset symbols to target allocation ratios (should sum to 1)
            total_value: Total initial portfolio value in quote currency
            prices: Dict mapping asset symbols to current prices
        """
        # Validate allocations
        alloc_sum = sum(allocations.values())
        if not (0.99 <= alloc_sum <= 1.01):
            raise ValueError(f"Asset allocations must sum to approximately 1.0, got {alloc_sum}")

        # Initialize assets based on allocations
        assets = {}

        # Calculate amount of each asset
        for symbol, allocation in allocations.items():
            if symbol not in prices or prices[symbol] <= 0:
                continue

            asset_value = allocation * total_value
            asset_amount = asset_value / prices[symbol]
            assets[symbol] = asset_amount

        # Calculate remaining quote currency
        allocated_value = sum(assets.get(s, 0) * prices.get(s, 0) for s in assets)
        quote_amount = total_value - allocated_value

        # Initialize with calculated values
        super().__init__(
            assets=assets,
            quote=quote_amount,
            interest_assets={},
            interest_quote=0.0
        )


class TargetPortfolio(Portfolio):
    """
    A Portfolio initialized to a specific target position and value at a given price.
    Assumes a single base instrument vs quote currency setup.
    Inherits from Portfolio.
    """
    def __init__(self, position: float, value: float, price: float):
        """
        Initializes the TargetPortfolio.

        Args:
            position: The target initial position ratio (0 to 1, relative to base instrument).
            value: The target initial total portfolio value (in quote currency).
            price: The price of the base instrument (in quote currency) used to calculate initial amounts.
        """
        # Calculate initial base and quote based on target position and value
        initial_base_value = position * value
        initial_base = initial_base_value / price if price != 0 else 0
        initial_quote = (1 - position) * value
        super().__init__(
            base=initial_base,
            quote=initial_quote,
            interest_base=0, # Starts with no debt/interest
            interest_quote=0
        )
