import Testing
import Foundation
import LocalLLMClientCore
import LocalLLMClientMLX
import LocalLLMClientTestUtilities

extension ModelTests {
    @Suite(.serialized, .timeLimit(.minutes(5)))
    struct LLMSessionMLXTests {

        private static func makeGeneralModel(size: LocalLLMClient.ModelSize = .default) -> LLMSession.DownloadModel {
            let modelId = LocalLLMClient.modelInfo(for: .general, modelSize: size)
            return .mlx(id: modelId)
        }

        private static func makeToolModel(size: LocalLLMClient.ModelSize = .default) -> LLMSession.DownloadModel {
            let modelId = LocalLLMClient.modelInfo(for: .tool, modelSize: size)
            return .mlx(id: modelId)
        }

        /// Helper: Runs a query and returns the output and all test tool instances for assertion
        private func runPortfolioQuery(_ query: String) async throws -> (
            output: String,
            holdingsTool: GetHoldingsTool,
            transactionsTool: GetTransactionsTool,
            portfolioTool: GetPortfolioValTool
        ) {
            let data = Data(mockData.utf8)
            let container = try! JSONDecoder().decode(MockDataContainer.self, from: data)
            let holdingsTool = GetHoldingsTool(provider: { container.holdings })
            let transactionsTool = GetTransactionsTool(provider: { container.transactions })
            let portfolioTool = GetPortfolioValTool(provider: { container.portfolio_value })

            let session = LLMSession(
                model: Self.makeToolModel(),
                messages: [.system(sysPrompt)],
                tools: [holdingsTool, transactionsTool, portfolioTool]
            )
            let output = try await session.respond(to: query)
            return (output, holdingsTool, transactionsTool, portfolioTool)
        }

        // === GROUP: SUMMARY & OVERVIEW TESTS ===

        @Test(.timeLimit(.minutes(2)))
        func testPortfolioSummaryWithSysPrompt() async throws {
            let (output, holdingsTool, transactionsTool, portfolioTool) = try await runPortfolioQuery("Can you summarize my portfolio performance?")
            print("Response:", output)
            #expect(output.contains("portfolio") || output.contains("holdings") || output.contains("market"))
            #expect(holdingsTool.invocationCount > 0)
            #expect(transactionsTool.invocationCount > 0)
            #expect(portfolioTool.invocationCount > 0)
        }

        @Test(.timeLimit(.minutes(2)))
        func testFullPortfolioOverview() async throws {
            let (output, holdingsTool, transactionsTool, portfolioTool) = try await runPortfolioQuery("Give me a full portfolio overview.")
            print("Full Portfolio Overview:\n\(output)")
            #expect(!output.isEmpty, "Should return full portfolio overview with holdings, trend, and transactions")
            #expect(holdingsTool.invocationCount > 0)
            #expect(transactionsTool.invocationCount > 0)
            #expect(portfolioTool.invocationCount > 0)
        }

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
//
//        // === GROUP: PORTFOLIO VALUE TESTS ===
//
//        @Test(.timeLimit(.minutes(2)))
//        func testTotalMarketValue() async throws {
//            let (output, holdingsTool, transactionsTool, portfolioTool) = try await runPortfolioQuery("Calculate total market value across all accounts.")
//            print("Total Market Value:\n\(output)")
//            #expect(!output.isEmpty, "Should return total market value")
//            #expect(portfolioTool.invocationCount > 0)
//            #expect(holdingsTool.invocationCount == 0)
//            #expect(transactionsTool.invocationCount == 0)
//        }

//        @Test(.timeLimit(.minutes(2)))
//        func testPortfolioTrend() async throws {
//            let (output, _, _, portfolioTool) = try await runPortfolioQuery("Show me the trend in my portfolio value.")
//            print("Portfolio Trend:\n\(output)")
//            #expect(!output.isEmpty, "Should return portfolio value trend")
//            #expect(portfolioTool.invocationCount > 0)
//        }
//
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
    }
}
