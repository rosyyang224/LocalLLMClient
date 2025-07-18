//
//  PortfolioValue.swift
//  LocalLLMClientExample
//
//  Created by Rosemary Yang on 7/17/25.
//

struct PortfolioValue: Codable {
    let clientID: String
    let marketChange: Double
    let marketValue: Double
    let valueDate: String
    let yearToDateRateOfReturnCumulative: Double
    let indices: [String]
    let contributionAndWithdraw: Double
    let yearToDateOfReturn: Double
    let growthCumulativeValueDate: String
    let netARR: Double
    let cumulativeARR: Double
}
