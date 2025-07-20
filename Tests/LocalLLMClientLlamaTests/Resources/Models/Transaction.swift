//
//  Transaction.swift
//  LocalLLMClientExample
//
//  Created by Rosemary Yang on 7/17/25.
//


struct Transaction: Codable {
    let cusip: String
    let description: String
    let transactiontype: String
    let transactiontypedesc: String
    let settlementdate: String
    let transactiondate: String
    let costprice: Double
    let cumulativeamount: Double
    let transactionamt: Double
    let sharesoffacevalue: Double
    let principal: Double
    let interest: Double
    let commission: Double
    let taxwithheld: Double
    let otherexpensesm: Double
    let account: String
    let stccy: String
    let uuid: String
}
