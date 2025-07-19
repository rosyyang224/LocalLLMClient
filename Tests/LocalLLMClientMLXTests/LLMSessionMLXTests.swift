import Testing
import Foundation
import LocalLLMClient
import LocalLLMClientCore
import LocalLLMClientMLX
import LocalLLMClientTestUtilities

func makeDownloadModel(model: LLMModel) -> LLMSession.DownloadModel {
    print("🔧 makeDownloadModel called with: \(model.name)")
    return .mlx(id: model.id)
}

func runPortfolioQuery(
    _ query: String,
    model: LLMModel = .qwen3
) async throws -> (
    output: String,
    holdingsTool: GetHoldingsTool,
    transactionsTool: GetTransactionsTool,
    portfolioTool: GetPortfolioValTool
) {
    print("🔧 Starting runPortfolioQuery with model: \(model.name)")
    print("🔧 Query: \(query)")
    
    let data = Data(mockData.utf8)
    let container = try! JSONDecoder().decode(MockDataContainer.self, from: data)
    let holdingsTool = GetHoldingsTool(provider: { container.holdings })
    let transactionsTool = GetTransactionsTool(provider: { container.transactions })
    let portfolioTool = GetPortfolioValTool(provider: { container.portfolio_value })

    print("🔧 Creating download model...")
    let downloadModel = makeDownloadModel(model: model)
    
    print("🔧 Creating LLM session...")
    let session = LLMSession(
        model: downloadModel,
        messages: [.system(sysPrompt)],
        tools: [holdingsTool, transactionsTool, portfolioTool]
    )
    
    print("🔧 Calling session.respond...")
    let output = try await session.respond(to: query)
    print("🔧 Got response: \(output.prefix(100))...")
    
    return (output, holdingsTool, transactionsTool, portfolioTool)
}

extension ModelTests {
    @Suite
    struct DebugTests {
        @Test(.timeLimit(.minutes(1)))
        func testBasicSetup() async throws {
            print("🔍 Testing basic setup...")
            
            // Test that our enum works
            let model = LLMModel.qwen3
            print("🔍 Model: \(model.name), ID: \(model.id)")
            
            // Test mock data loading
            let data = Data(mockData.utf8)
            let container = try JSONDecoder().decode(MockDataContainer.self, from: data)
            print("🔍 Mock data loaded successfully")
            print("🔍 Holdings count: \(container.holdings.count)")
            print("🔍 Transactions count: \(container.transactions.count)")
            
            // Test tool creation
            let holdingsTool = GetHoldingsTool(provider: { container.holdings })
            let transactionsTool = GetTransactionsTool(provider: { container.transactions })
            let portfolioTool = GetPortfolioValTool(provider: { container.portfolio_value })
            print("🔍 Tools created successfully")
            
            print("🔍 ✅ Basic setup test passed")
        }
        
        @Test(.timeLimit(.minutes(1)))
        func testModelCreation() async throws {
            print("🔍 Testing model creation...")
            let model = LLMModel.qwen3
            let downloadModel = makeDownloadModel(model: model)
            print("🔍 Download model created: \(downloadModel)")
            print("🔍 ✅ Model creation test passed")
        }
    }

    @Suite
    struct HoldingsToolTests {
        @Test(.timeLimit(.minutes(1))) func testUSEquityHoldings() async throws {
            print("🧪 Testing US Equity Holdings...")
            let (output, h, t, p) = try await runPortfolioQuery("What are my US equity holdings?")
            #expect(h.invocationCount > 0)
            #expect(t.invocationCount == 0)
            #expect(p.invocationCount == 0)
            #expect(!output.isEmpty)
            print("🧪 ✅ US Equity Holdings test passed")
        }
        @Test(.timeLimit(.minutes(1))) func testFixedIncomeHoldings() async throws {
            print("🧪 Testing Fixed Income Holdings...")
            let (output, h, t, p) = try await runPortfolioQuery("List all fixed income holdings.")
            #expect(h.invocationCount > 0)
            #expect(t.invocationCount == 0)
            #expect(p.invocationCount == 0)
            #expect(!output.isEmpty)
            print("🧪 ✅ Fixed Income Holdings test passed")
        }
        @Test(.timeLimit(.minutes(1))) func testHoldingsByRegion() async throws {
            print("🧪 Testing Holdings by Region...")
            let (output, h, t, p) = try await runPortfolioQuery("Show my holdings in the United States.")
            #expect(h.invocationCount > 0)
            #expect(t.invocationCount == 0)
            #expect(p.invocationCount == 0)
            #expect(!output.isEmpty)
            print("🧪 ✅ Holdings by Region test passed")
        }
    }

    @Suite
    struct PortfolioToolTests {
        @Test(.timeLimit(.minutes(1))) func testTotalMarketValue() async throws {
            print("🧪 Testing Total Market Value...")
            let (output, h, t, p) = try await runPortfolioQuery("Calculate total market value across all accounts.")
            #expect(h.invocationCount == 0)
            #expect(t.invocationCount == 0)
            #expect(p.invocationCount > 0)
            #expect(!output.isEmpty)
            print("🧪 ✅ Total Market Value test passed")
        }
        @Test(.timeLimit(.minutes(1))) func testPortfolioTrend() async throws {
            print("🧪 Testing Portfolio Trend...")
            let (output, h, t, p) = try await runPortfolioQuery("Show me the trend in my portfolio value.")
            #expect(h.invocationCount == 0)
            #expect(t.invocationCount == 0)
            #expect(p.invocationCount > 0)
            #expect(!output.isEmpty)
            print("🧪 ✅ Portfolio Trend test passed")
        }
        @Test(.timeLimit(.minutes(1))) func testHighestPortfolioValue() async throws {
            print("🧪 Testing Highest Portfolio Value...")
            let (output, h, t, p) = try await runPortfolioQuery("What was my highest portfolio value?")
            #expect(h.invocationCount == 0)
            #expect(t.invocationCount == 0)
            #expect(p.invocationCount > 0)
            #expect(!output.isEmpty)
            print("🧪 ✅ Highest Portfolio Value test passed")
        }
    }
    
    @Suite
    struct TransactionToolTests {
        @Test(.timeLimit(.minutes(1))) func testRecentTransactions() async throws {
            print("🧪 Testing Recent Transactions...")
            let (output, h, t, p) = try await runPortfolioQuery("What are my most recent transactions?")
            #expect(h.invocationCount == 0)
            #expect(t.invocationCount > 0)
            #expect(p.invocationCount == 0)
            #expect(!output.isEmpty)
            print("🧪 ✅ Recent Transactions test passed")
        }
        @Test(.timeLimit(.minutes(1))) func testTransactionsByType() async throws {
            print("🧪 Testing Transactions by Type...")
            let (output, h, t, p) = try await runPortfolioQuery("Show me all BUY transactions.")
            #expect(h.invocationCount == 0)
            #expect(t.invocationCount > 0)
            #expect(p.invocationCount == 0)
            #expect(!output.isEmpty)
            print("🧪 ✅ Transactions by Type test passed")
        }
        @Test(.timeLimit(.minutes(1))) func testLargeTransactions() async throws {
            print("🧪 Testing Large Transactions...")
            let (output, h, t, p) = try await runPortfolioQuery("List all transactions greater than $10,000.")
            #expect(h.invocationCount == 0)
            #expect(t.invocationCount > 0)
            #expect(p.invocationCount == 0)
            #expect(!output.isEmpty)
            print("🧪 ✅ Large Transactions test passed")
        }
    }
    
    
    @Suite
    struct CrossDataToolTests {
        @Test(.timeLimit(.minutes(1))) func testMultiAspectQuery() async throws {
            print("🧪 Testing Multi-Aspect Query...")
            let (output, h, t, p) = try await runPortfolioQuery("Give me an overview of my holdings and recent transactions in the last month.")
            #expect(h.invocationCount > 0)
            #expect(t.invocationCount > 0)
            #expect(p.invocationCount == 0)
            #expect(!output.isEmpty)
            print("🧪 ✅ Multi-Aspect Query test passed")
        }
        @Test(.timeLimit(.minutes(1))) func testPercentFixedIncome() async throws {
            print("🧪 Testing Percent Fixed Income...")
            let (output, h, t, p) = try await runPortfolioQuery("What percent of my portfolio is in fixed income?")
            #expect(h.invocationCount > 0)
            #expect(t.invocationCount == 0)
            #expect(p.invocationCount == 0)
            #expect(!output.isEmpty)
            print("🧪 ✅ Percent Fixed Income test passed")
        }
        @Test(.timeLimit(.minutes(1))) func testPerformanceByAssetClass() async throws {
            print("🧪 Testing Performance by Asset Class...")
            let (output, h, t, p) = try await runPortfolioQuery("Compare equity vs fixed income returns.")
            #expect(h.invocationCount > 0)
            #expect(t.invocationCount == 0)
            #expect(p.invocationCount == 0)
            #expect(!output.isEmpty)
            print("🧪 ✅ Performance by Asset Class test passed")
        }
        @Test(.timeLimit(.minutes(1))) func testBreakdownIntlVsUS() async throws {
            print("🧪 Testing Breakdown Intl vs US...")
            let (output, h, t, p) = try await runPortfolioQuery("Break down my international vs US investments.")
            #expect(h.invocationCount > 0)
            #expect(t.invocationCount == 0)
            #expect(p.invocationCount == 0)
            #expect(!output.isEmpty)
            print("🧪 ✅ Breakdown Intl vs US test passed")
        }
    }
    
    @Suite
    struct SummarizationTests {
        @Test(.timeLimit(.minutes(1))) func testPortfolioSummary() async throws {
            print("🧪 Testing Portfolio Summary...")
            let (output, h, t, p) = try await runPortfolioQuery("Show me a summary of my portfolio")
            #expect(h.invocationCount > 0)
            #expect(t.invocationCount > 0)
            #expect(p.invocationCount > 0)
            #expect(!output.isEmpty)
            print("🧪 ✅ Portfolio Summary test passed")
        }
        @Test(.timeLimit(.minutes(1))) func testFullPortfolioOverview() async throws {
            print("🧪 Testing Full Portfolio Overview...")
            let (output, h, t, p) = try await runPortfolioQuery("Give me a full portfolio overview.")
            #expect(h.invocationCount > 0)
            #expect(t.invocationCount > 0)
            #expect(p.invocationCount > 0)
            #expect(!output.isEmpty)
            print("🧪 ✅ Full Portfolio Overview test passed")
        }
    }
    
    @Suite(.serialized, .timeLimit(.minutes(30)))
    struct AllToolTestSweep {
        @Test
        func testAllMLXModels() async throws {
            print("🚀 Starting testAllMLXModels")
            
            let allMLXModels: [LLMModel] = LLMModel.allCases.filter { $0.isMLX && $0.supportsTools }
            print("🚀 Found \(allMLXModels.count) MLX models that support tools:")
            for model in allMLXModels {
                print("  - \(model.name)")
            }
            
            let allTestCases: [(query: String, expectHoldings: Bool, expectTransactions: Bool, expectPortfolio: Bool)] = [
                // HoldingsToolTests
                ("What are my US equity holdings?", true, false, false),
                ("List all fixed income holdings.", true, false, false),
                ("Show my holdings in the United States.", true, false, false),
                
                // PortfolioToolTests
                ("Calculate total market value across all accounts.", false, false, true),
                ("Show me the trend in my portfolio value.", false, false, true),
                ("What was my highest portfolio value?", false, false, true),
                
                // TransactionToolTests
                ("What are my most recent transactions?", false, true, false),
                ("Show me all BUY transactions.", false, true, false),
                ("List all transactions greater than $10,000.", false, true, false),
                
                // CrossDataToolTests
                ("Give me an overview of my holdings and recent transactions in the last month.", true, true, false),
                ("What percent of my portfolio is in fixed income?", true, false, false),
                ("Compare equity vs fixed income returns.", true, false, false),
                ("Break down my international vs US investments.", true, false, false),
                
                // SummarizationTests
                ("Show me a summary of my portfolio", true, true, true),
                ("Give me a full portfolio overview.", true, true, true)
            ]
            print("🚀 Testing \(allTestCases.count) test cases")

            for (modelIndex, model) in allMLXModels.enumerated() {
                print("🚀 Testing model \(modelIndex + 1)/\(allMLXModels.count): \(model.name)")
                
                for (testIndex, (query, expectHoldings, expectTransactions, expectPortfolio)) in allTestCases.enumerated() {
                    print("🚀   Test \(testIndex + 1)/\(allTestCases.count): \(query)")
                    
                    do {
                        let (output, h, t, p) = try await runPortfolioQuery(query, model: model)
                        
                        // Tool call checks
                        if expectHoldings {
                            #expect(h.invocationCount > 0, "Expected holdings tool to be called")
                        } else {
                            #expect(h.invocationCount == 0, "Expected holdings tool NOT to be called")
                        }
                        if expectTransactions {
                            #expect(t.invocationCount > 0, "Expected transactions tool to be called")
                        } else {
                            #expect(t.invocationCount == 0, "Expected transactions tool NOT to be called")
                        }
                        if expectPortfolio {
                            #expect(p.invocationCount > 0, "Expected portfolio tool to be called")
                        } else {
                            #expect(p.invocationCount == 0, "Expected portfolio tool NOT to be called")
                        }
                        #expect(!output.isEmpty, "Expected non-empty output")
                        
                        print("🚀   ✅ Test passed for \(model.name)")
                    } catch {
                        print("🚀   ❌ Test failed for \(model.name): \(error)")
                        throw error
                    }
                }
            }
            print("🚀 All tests completed successfully!")
        }
    }
    
//    @Suite(.serialized, .timeLimit(.minutes(5)))
//    struct LLMSessionMLXTests {
//
//        private static func makeGeneralModel(size: LocalLLMClient.ModelSize = .default) -> LLMSession.DownloadModel {
//            let modelId = LocalLLMClient.modelInfo(for: .general, modelSize: size)
//            return .mlx(id: modelId)
//        }
//
//        private static func makeToolModel(size: LocalLLMClient.ModelSize = .default) -> LLMSession.DownloadModel {
//            let modelId = LocalLLMClient.modelInfo(for: .tool, modelSize: size)
//            return .mlx(id: modelId)
//        }
//
//        /// Helper: Runs a query and returns the output and all test tool instances for assertion
//        private func runPortfolioQuery(_ query: String) async throws -> (
//            output: String,
//            holdingsTool: GetHoldingsTool,
//            transactionsTool: GetTransactionsTool,
//            portfolioTool: GetPortfolioValTool
//        ) {
//            let data = Data(mockData.utf8)
//            let container = try! JSONDecoder().decode(MockDataContainer.self, from: data)
//            let holdingsTool = GetHoldingsTool(provider: { container.holdings })
//            let transactionsTool = GetTransactionsTool(provider: { container.transactions })
//            let portfolioTool = GetPortfolioValTool(provider: { container.portfolio_value })
//
//            let session = LLMSession(
//                model: Self.makeToolModel(),
//                messages: [.system(sysPrompt)],
//                tools: [holdingsTool, transactionsTool, portfolioTool]
//            )
//            let output = try await session.respond(to: query)
//            return (output, holdingsTool, transactionsTool, portfolioTool)
//        }
//
//        // === GROUP: SUMMARY & OVERVIEW TESTS ===
//
//        @Test(.timeLimit(.minutes(2)))
//        func testPortfolioSummaryWithSysPrompt() async throws {
//            let (output, holdingsTool, transactionsTool, portfolioTool) = try await runPortfolioQuery("Can you summarize my portfolio performance?")
//            print("Response:", output)
//            #expect(output.contains("portfolio") || output.contains("holdings") || output.contains("market"))
//            #expect(holdingsTool.invocationCount > 0)
//            #expect(transactionsTool.invocationCount > 0)
//            #expect(portfolioTool.invocationCount > 0)
//        }
//
//        @Test(.timeLimit(.minutes(2)))
//        func testFullPortfolioOverview() async throws {
//            let (output, holdingsTool, transactionsTool, portfolioTool) = try await runPortfolioQuery("Give me a full portfolio overview.")
//            print("Full Portfolio Overview:\n\(output)")
//            #expect(!output.isEmpty, "Should return full portfolio overview with holdings, trend, and transactions")
//            #expect(holdingsTool.invocationCount > 0)
//            #expect(transactionsTool.invocationCount > 0)
//            #expect(portfolioTool.invocationCount > 0)
//        }
//
//        // === GROUP: HOLDINGS TESTS ===
//
//        @Test(.timeLimit(.minutes(2)))
//        func testUSEquityHoldings() async throws {
//            let (output, holdingsTool, transactionsTool, portfolioTool) = try await runPortfolioQuery("What are my US equity holdings?")
//            print("US Equity Holdings:\n\(output)")
//            #expect(!output.isEmpty, "Should return US equity holdings")
//            #expect(holdingsTool.invocationCount > 0)
//            #expect(transactionsTool.invocationCount == 0)
//            #expect(portfolioTool.invocationCount == 0)
//        }
//
//        @Test(.timeLimit(.minutes(2)))
//        func testFixedIncomeHoldings() async throws {
//            let (output, holdingsTool, transactionsTool, portfolioTool) = try await runPortfolioQuery("List all fixed income holdings.")
//            print("Fixed Income Holdings:\n\(output)")
//            #expect(!output.isEmpty, "Should return fixed income holdings")
//            #expect(holdingsTool.invocationCount > 0)
//            #expect(transactionsTool.invocationCount == 0)
//            #expect(portfolioTool.invocationCount == 0)
//        }
//
//        @Test(.timeLimit(.minutes(2)))
//        func testHoldingsByRegion() async throws {
//            let (output, holdingsTool, _, _) = try await runPortfolioQuery("Show my holdings in the United States.")
//            print("Holdings by Region:\n\(output)")
//            #expect(!output.isEmpty, "Should return holdings for United States")
//            #expect(holdingsTool.invocationCount > 0)
//        }
//
//        // === GROUP: TRANSACTIONS TESTS ===
//
//        @Test(.timeLimit(.minutes(2)))
//        func testRecentTransactions() async throws {
//            let (output, holdingsTool, transactionsTool, portfolioTool) = try await runPortfolioQuery("What are my most recent transactions?")
//            print("Recent Transactions:\n\(output)")
//            #expect(!output.isEmpty, "Should return recent transactions")
//            #expect(transactionsTool.invocationCount > 0)
//            #expect(holdingsTool.invocationCount == 0)
//            #expect(portfolioTool.invocationCount == 0)
//        }
//
//        @Test(.timeLimit(.minutes(2)))
//        func testTransactionsByType() async throws {
//            let (output, _, transactionsTool, _) = try await runPortfolioQuery("Show me all BUY transactions.")
//            print("Transactions by Type:\n\(output)")
//            #expect(!output.isEmpty, "Should return BUY transactions")
//            #expect(transactionsTool.invocationCount > 0)
//        }
//
//        @Test(.timeLimit(.minutes(2)))
//        func testLargeTransactions() async throws {
//            let (output, _, transactionsTool, _) = try await runPortfolioQuery("List all transactions greater than $10,000.")
//            print("Large Transactions:\n\(output)")
//            #expect(!output.isEmpty, "Should return large transactions")
//            #expect(transactionsTool.invocationCount > 0)
//        }
////
////        // === GROUP: PORTFOLIO VALUE TESTS ===
////
//        @Test(.timeLimit(.minutes(2)))
//        func testTotalMarketValue() async throws {
//            let (output, holdingsTool, transactionsTool, portfolioTool) = try await runPortfolioQuery("Calculate total market value across all accounts.")
//            print("Total Market Value:\n\(output)")
//            #expect(!output.isEmpty, "Should return total market value")
//            #expect(portfolioTool.invocationCount > 0)
//            #expect(holdingsTool.invocationCount == 0)
//            #expect(transactionsTool.invocationCount == 0)
//        }
//
//        @Test(.timeLimit(.minutes(2)))
//        func testPortfolioTrend() async throws {
//            let (output, _, _, portfolioTool) = try await runPortfolioQuery("Show me the trend in my portfolio value.")
//            print("Portfolio Trend:\n\(output)")
//            #expect(!output.isEmpty, "Should return portfolio value trend")
//            #expect(portfolioTool.invocationCount > 0)
//        }
////
//        @Test(.timeLimit(.minutes(2)))
//        func testHighestPortfolioValue() async throws {
//            let (output, _, _, portfolioTool) = try await runPortfolioQuery("What was my highest portfolio value?")
//            print("Highest Portfolio Value:\n\(output)")
//            #expect(!output.isEmpty, "Should return highest portfolio value")
//            #expect(portfolioTool.invocationCount > 0)
//        }
//
//        // === GROUP: MIXED/FULL COVERAGE TESTS ===
//
//        @Test(.timeLimit(.minutes(2)))
//        func testMultiAspectQuery() async throws {
//            let (output, holdingsTool, transactionsTool, portfolioTool) = try await runPortfolioQuery("Give me an overview of my holdings and recent transactions in the last month.")
//            print("Multi-Aspect Overview:\n\(output)")
//            #expect(!output.isEmpty, "Should return overview with holdings and recent transactions")
//            #expect(holdingsTool.invocationCount > 0)
//            #expect(transactionsTool.invocationCount > 0)
//        }
//    }
}
