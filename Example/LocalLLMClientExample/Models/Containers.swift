//
//  MockDataContainer.swift
//  LocalLLMClientExample
//
//  Created by Rosemary Yang on 7/17/25.
//

struct MockDataContainer: Codable {
    let portfolio_value: [PortfolioValue]
    let transactions: [Transaction]
    let holdings: [Holding]
    
    init() {
        holdings = []
        portfolio_value = []
        transactions = []
    }
}
