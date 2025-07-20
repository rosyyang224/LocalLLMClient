//
//  GetPortfolioValTool.swift
//  LocalLLMClientExample
//
//  Created by Rosemary Yang on 7/17/25.
//

import Foundation
import LocalLLMClient
import LocalLLMClientMacros

private func effectiveFilter(_ value: String?) -> String? {
    return (value == "all") ? nil : value
}

@Tool("get_portfolio_value")
struct GetPortfolioValTool {
    let description = "Query your portfolio value snapshots. Filter by date range or index, or retrieve summary statistics like highest, lowest, and trend over time."

    @ToolArguments
    struct Arguments {
        @ToolArgument("Start date (inclusive, format YYYY-MM-DD).")
        var startDate: String?
        @ToolArgument("End date (inclusive, format YYYY-MM-DD).")
        var endDate: String?
        @ToolArgument("Filter for a specific market index (e.g. 'S&P 500').")
        var index: String?
        @ToolArgument("Return summary: 'highest', 'lowest', 'trend', or leave blank for raw results.")
        var summary: String?
    }

    let portfolioValProvider: @Sendable () -> [PortfolioValue]

    func call(arguments: Arguments) async throws -> ToolOutput {
        print("[GetPortfolioValTool] called with arguments:")
        print("  startDate: \(arguments.startDate ?? "nil")")
        print("  endDate: \(arguments.endDate ?? "nil")")
        print("  index: \(arguments.index ?? "nil")")
        print("  summary: \(arguments.summary ?? "nil")")

        let all = portfolioValProvider()
        print("[GetPortfolioValTool] total portfolio values: \(all.count)")

        let filtered = all.filter { pv in
            if let idx = effectiveFilter(arguments.index), !pv.indices.contains(where: { $0.localizedCaseInsensitiveContains(idx) }) {
                return false
            }
            if let start = effectiveFilter(arguments.startDate), pv.valueDate < start {
                return false
            }
            if let end = effectiveFilter(arguments.endDate), pv.valueDate > end {
                return false
            }
            return true
        }

        print("[GetPortfolioValTool] filtered values count: \(filtered.count)")

        if let summary = arguments.summary?.lowercased() {
            switch summary {
            case "highest":
                if let maxPV = filtered.max(by: { $0.marketValue < $1.marketValue }) {
                    print("[GetPortfolioValTool] highest found: \(maxPV)")
                    return ToolOutput(data: ["type": "highest", "portfolio_value": maxPV])
                }
            case "lowest":
                if let minPV = filtered.min(by: { $0.marketValue < $1.marketValue }) {
                    print("[GetPortfolioValTool] lowest found: \(minPV)")
                    return ToolOutput(data: ["type": "lowest", "portfolio_value": minPV])
                }
            case "trend":
                let points = filtered
                    .sorted(by: { $0.valueDate < $1.valueDate })
                    .map { ["date": $0.valueDate, "marketValue": $0.marketValue] }
                print("[GetPortfolioValTool] trend points count: \(points.count)")
                return ToolOutput(data: ["type": "trend", "points": points])
            case "latest":
                print("[GetPortfolioValTool] 'latest' treated as raw data request")
                break // Falls through to return raw filtered data
            default:
                print("[GetPortfolioValTool] Unknown summary type: \(summary)")
                break
            }
        }

        print("[GetPortfolioValTool] returning raw filtered data")
        return ToolOutput(data: [
            "portfolio_values": filtered
        ])
    }
}
