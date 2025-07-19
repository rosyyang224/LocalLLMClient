//
//  Holding.swift
//  LocalLLMClientExample
//
//  Created by Rosemary Yang on 7/17/25.
//

struct Holding: Codable {
    let symbol: String
    let cusip: String
    let fxrate: Double
    let marketvalueinbccy: Double
    let totalmarketvalue: Double
    let assetclass: String
    let uuid: String
    let costpricesccy: Double
    let accrualsymbol: String
    let totalcostinbccy: Double
    let maturitydate: String?
    let assettemplatetype: String
    let marketyield: Double?
    let accruedcashvaluesccy: Double
    let marketpricesccy: Double
    let marketplinsccy: Double
    let forexpl: Double
    let accounttype: String
    let countryregion: String
    let marketplinbccy: Double
    let marketplpercentinsccy: Double
    let accrualtype: String
    let totalmarketvaluesccy: Double
    let totalmarketvalueccy: Double
    let ytm: Double?
    let securitytype: String
    let accrualsccy: Double
    let fxmarket: Double
    let sccy: String
}
