//
//  Utils.swift
//  LocalLLMClientExample
//
//  Created by Rosemary Yang on 7/18/25.
//

import Foundation
import LocalLLMClient
import LocalLLMClientMLX
import LocalLLMClientLlama

func loadMockDataContainer(from jsonString: String) -> MockDataContainer? {
    let data = Data(jsonString.utf8)
    let decoder = JSONDecoder()
    do {
        return try decoder.decode(MockDataContainer.self, from: data)
    } catch {
        print("Failed to decode mock data: \(error)")
        return nil
    }
}

func makeLLMTools(container: MockDataContainer) -> [any LLMTool] {
    let holdings = container.holdings
    let portfolio_vals = container.portfolio_value
    let transactions = container.transactions

    let getHoldingsTool = GetHoldingsTool(holdingsProvider: { holdings })
    let getPortfolioValTool = GetPortfolioValTool(portfolioValProvider: { portfolio_vals })
    let getTransactionsTool = GetTransactionsTool(transactionsProvider: { transactions })

    return [
        getHoldingsTool,
        getPortfolioValTool,
        getTransactionsTool,
    ]
}
