import Foundation
import LocalLLMClientCore
import LocalLLMClientMacros
import LocalLLMClientUtility

// MARK: - Holdings Tool

public final class GetHoldingsTool: LLMTool {
    public let name = "get_holdings"
    public let description = "Retrieve portfolio holdings, filterable by symbol, asset class, region, account type, profit/loss, or value."

    private let _invocationCount = Locked(0)
    private let _lastArguments = Locked<Arguments?>(nil)
    private let holdingsProvider: @Sendable () -> [Holding]

    init(provider: @escaping @Sendable () -> [Holding]) {
        self.holdingsProvider = provider
    }

    public var invocationCount: Int {
        _invocationCount.withLock { $0 }
    }

    public var lastArguments: Arguments? {
        _lastArguments.withLock { $0 }
    }

    @ToolArguments
    public struct Arguments: Sendable {
        @ToolArgument("The security symbol (e.g. 'AAPL').")
        public var symbol: String?
        @ToolArgument("Asset class (e.g. 'Equity', 'Fixed Income').")
        public var assetclass: String?
        @ToolArgument("Country or region (e.g. 'United States', 'Hong Kong').")
        public var countryregion: String?
        @ToolArgument("Account type (e.g. 'Brokerage', 'Retirement').")
        public var accounttype: String?
        @ToolArgument("Only holdings with profit/loss (in settlement currency) >= this value.")
        public var min_marketplinsccy: Double?
        @ToolArgument("Only holdings with profit/loss (in settlement currency) <= this value.")
        public var max_marketplinsccy: Double?
        @ToolArgument("Only holdings with market value (in base currency) >= this value.")
        public var min_marketvalueinbccy: Double?
        @ToolArgument("Only holdings with market value (in base currency) <= this value.")
        public var max_marketvalueinbccy: Double?
    }

    public func call(arguments: Arguments) async throws -> ToolOutput {
        _invocationCount.withLock { $0 += 1 }
        _lastArguments.withLock { $0 = arguments }

        let all = holdingsProvider()
        let filtered = all.filter { h in
            if let v = arguments.symbol, !h.symbol.localizedCaseInsensitiveContains(v) { return false }
            if let v = arguments.assetclass, !h.assetclass.localizedCaseInsensitiveContains(v) { return false }
            if let v = arguments.countryregion, !h.countryregion.localizedCaseInsensitiveContains(v) { return false }
            if let v = arguments.accounttype, !h.accounttype.localizedCaseInsensitiveContains(v) { return false }
            if let minPL = arguments.min_marketplinsccy, h.marketplinsccy < minPL { return false }
            if let maxPL = arguments.max_marketplinsccy, h.marketplinsccy > maxPL { return false }
            if let minVal = arguments.min_marketvalueinbccy, h.marketvalueinbccy < minVal { return false }
            if let maxVal = arguments.max_marketvalueinbccy, h.marketvalueinbccy > maxVal { return false }
            return true
        }

        return ToolOutput(data: [
            "holdings": filtered
        ])
    }

    public func reset() {
        _invocationCount.withLock { $0 = 0 }
        _lastArguments.withLock { $0 = nil }
    }
}

// MARK: - Transactions Tool

public final class GetTransactionsTool: LLMTool {
    public let name = "get_transactions"
    public let description = "Retrieve and filter your transaction history by symbol (CUSIP), transaction type, account, date range, or amount."

    private let _invocationCount = Locked(0)
    private let _lastArguments = Locked<Arguments?>(nil)
    private let transactionsProvider: @Sendable () -> [Transaction]

    init(provider: @escaping @Sendable () -> [Transaction]) {
        self.transactionsProvider = provider
    }

    public var invocationCount: Int {
        _invocationCount.withLock { $0 }
    }

    public var lastArguments: Arguments? {
        _lastArguments.withLock { $0 }
    }

    @ToolArguments
    public struct Arguments: Sendable {
        @ToolArgument("CUSIP or partial CUSIP of the security.")
        public var cusip: String?
        @ToolArgument("Transaction type (e.g. 'BUY', 'SELL').")
        public var transactiontype: String?
        @ToolArgument("Account name (e.g. 'Brokerage Account 1').")
        public var account: String?
        @ToolArgument("Start date (YYYY-MM-DD, inclusive).")
        public var startDate: String?
        @ToolArgument("End date (YYYY-MM-DD, inclusive).")
        public var endDate: String?
        @ToolArgument("Minimum transaction amount.")
        public var minTransactionAmt: Double?
        @ToolArgument("Maximum transaction amount.")
        public var maxTransactionAmt: Double?
    }

    public func call(arguments: Arguments) async throws -> ToolOutput {
        _invocationCount.withLock { $0 += 1 }
        _lastArguments.withLock { $0 = arguments }

        let all = transactionsProvider()
        let filtered = all.filter { txn in
            if let v = arguments.cusip, !txn.cusip.localizedCaseInsensitiveContains(v) { return false }
            if let v = arguments.transactiontype, !txn.transactiontype.localizedCaseInsensitiveContains(v) { return false }
            if let v = arguments.account, !txn.account.localizedCaseInsensitiveContains(v) { return false }
            if let minAmt = arguments.minTransactionAmt, txn.transactionamt < minAmt { return false }
            if let maxAmt = arguments.maxTransactionAmt, txn.transactionamt > maxAmt { return false }
            if let start = arguments.startDate, txn.transactiondate < start { return false }
            if let end = arguments.endDate, txn.transactiondate > end { return false }
            return true
        }

        return ToolOutput(data: [
            "transactions": filtered
        ])
    }

    public func reset() {
        _invocationCount.withLock { $0 = 0 }
        _lastArguments.withLock { $0 = nil }
    }
}

// MARK: - Portfolio Value Tool

public final class GetPortfolioValTool: LLMTool {
    public let name = "get_portfolio_value"
    public let description = "Query your portfolio value snapshots. Filter by date range or index, or retrieve summary statistics like highest, lowest, and trend over time."

    private let _invocationCount = Locked(0)
    private let _lastArguments = Locked<Arguments?>(nil)
    private let portfolioValProvider: @Sendable () -> [PortfolioValue]

    init(provider: @escaping @Sendable () -> [PortfolioValue]) {
        self.portfolioValProvider = provider
    }

    public var invocationCount: Int {
        _invocationCount.withLock { $0 }
    }

    public var lastArguments: Arguments? {
        _lastArguments.withLock { $0 }
    }

    @ToolArguments
    public struct Arguments: Sendable {
        @ToolArgument("Start date (inclusive, format YYYY-MM-DD).")
        public var startDate: String?
        @ToolArgument("End date (inclusive, format YYYY-MM-DD).")
        public var endDate: String?
        @ToolArgument("Filter for a specific market index (e.g. 'S&P 500').")
        public var index: String?
        @ToolArgument("Return summary: 'highest', 'lowest', 'trend', or leave blank for raw results.")
        public var summary: String?
    }

    public func call(arguments: Arguments) async throws -> ToolOutput {
        _invocationCount.withLock { $0 += 1 }
        _lastArguments.withLock { $0 = arguments }

        let all = portfolioValProvider()
        let filtered = all.filter { pv in
            if let idx = arguments.index, !pv.indices.contains(where: { $0.localizedCaseInsensitiveContains(idx) }) {
                return false
            }
            if let start = arguments.startDate, pv.valueDate < start {
                return false
            }
            if let end = arguments.endDate, pv.valueDate > end {
                return false
            }
            return true
        }

        if let summary = arguments.summary?.lowercased() {
            switch summary {
            case "highest":
                if let maxPV = filtered.max(by: { $0.marketValue < $1.marketValue }) {
                    return ToolOutput(data: ["type": "highest", "portfolio_value": maxPV])
                }
            case "lowest":
                if let minPV = filtered.min(by: { $0.marketValue < $1.marketValue }) {
                    return ToolOutput(data: ["type": "lowest", "portfolio_value": minPV])
                }
            case "trend":
                let points = filtered
                    .sorted(by: { $0.valueDate < $1.valueDate })
                    .map { ["date": $0.valueDate as Sendable, "marketValue": $0.marketValue as Sendable] }
                return ToolOutput(data: ["type": "trend", "points": points])
            default:
                break
            }
        }

        return ToolOutput(data: [
            "portfolio_values": filtered
        ])
    }

    public func reset() {
        _invocationCount.withLock { $0 = 0 }
        _lastArguments.withLock { $0 = nil }
    }
}
