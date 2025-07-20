import Testing
import Foundation
import LocalLLMClient
import LocalLLMClientCore
import LocalLLMClientLlama
import LocalLLMClientTestUtilities

extension ModelTests {
    @Suite(.serialized, .timeLimit(.minutes(5)))
    struct LLMSessionLlamaTests {
        
        // MARK: - Model Creation Functions
        
        public static func makeToolModel(model: LLMModel = .phi4mini) -> LLMSession.DownloadModel {
            print("Using TOOL model: \(model.name) - \(model.id)")
            
            switch model {
            case .phi4mini:
                return .llama(
                    id: "unsloth/Phi-4-mini-instruct-GGUF",
                    model: "Phi-4-mini-instruct-Q4_K_M.gguf",
                    mmproj: nil,
                    parameter: .init(
                        context: 10240,
                        temperature: 0.7,
                        topK: 40,
                        topP: 0.9
                    )
                )
            case .gemma3_4b:
                return .llama(
                    id: "lmstudio-community/gemma-3-4B-it-qat-GGUF",
                    model: "gemma-3-4B-it-QAT-Q4_0.gguf",
                    mmproj: nil,
                    parameter: .init(
                        context: 10240,
                        temperature: 0.7,
                        topK: 40,
                        topP: 0.9
                    )
                )
            default:
                fatalError("Unsupported model for Llama tests: \(model)")
            }
        }
        
        // MARK: - Inline Mock Data and Tool Creation
        
        private static func createMockPortfolioTools() -> (GetHoldingsTool, GetTransactionsTool, GetPortfolioValTool) {
            print("Creating mock portfolio tools...")
            
            let data = Data(mockData.utf8)
            let decoder = JSONDecoder()
            
            guard let container = try? decoder.decode(MockDataContainer.self, from: data) else {
                fatalError("Failed to decode mock data")
            }
            
            let holdingsTool = GetHoldingsTool(provider: { container.holdings })
            let transactionsTool = GetTransactionsTool(provider: { container.transactions })
            let portfolioTool = GetPortfolioValTool(provider: { container.portfolio_value })
            
            print("Created tools: Holdings(\(container.holdings.count) items), Transactions(\(container.transactions.count) items), Portfolio(\(container.portfolio_value.count) items)")
            
            return (holdingsTool, transactionsTool, portfolioTool)
        }
        
        // MARK: - Portfolio Tool Tests
        
        @Test
        func portfolioHoldingsToolCall() async throws {
            let (holdingsTool, transactionsTool, portfolioTool) = Self.createMockPortfolioTools()
            
            let session = LLMSession(
                model: Self.makeToolModel(),
                messages: [.system(createSystemPrompt())],
                tools: [holdingsTool, transactionsTool, portfolioTool]
            )
            
            let response = try await session.respond(to: "What are my US equity holdings?")
            print("Holdings response: \(response)")
            
            #expect(holdingsTool.invocationCount > 0, "Holdings tool should have been called")
            #expect(transactionsTool.invocationCount == 0, "Transactions tool should not have been called")
            #expect(portfolioTool.invocationCount == 0, "Portfolio tool should not have been called")
            #expect(!response.isEmpty, "Response should not be empty")
        }
        
        @Test
        func portfolioTransactionsToolCall() async throws {
            let (holdingsTool, transactionsTool, portfolioTool) = Self.createMockPortfolioTools()
            
            let session = LLMSession(
                model: Self.makeToolModel(),
                messages: [.system(createSystemPrompt())],
                tools: [holdingsTool, transactionsTool, portfolioTool]
            )
            
            let response = try await session.respond(to: "Show me my buy transactions")
            print("Transactions response: \(response)")
            
            #expect(holdingsTool.invocationCount == 0, "Holdings tool should not have been called")
            #expect(transactionsTool.invocationCount > 0, "Transactions tool should have been called")
            #expect(portfolioTool.invocationCount == 0, "Portfolio tool should not have been called")
            #expect(!response.isEmpty, "Response should not be empty")
        }
        
        @Test
        func portfolioValueToolCall() async throws {
            let (holdingsTool, transactionsTool, portfolioTool) = Self.createMockPortfolioTools()
            
            let session = LLMSession(
                model: Self.makeToolModel(),
                messages: [.system(createSystemPrompt())],
                tools: [holdingsTool, transactionsTool, portfolioTool]
            )
            
            let response = try await session.respond(to: "What was my portfolio value in August 2024?")
            print("Portfolio value response: \(response)")
            
            #expect(holdingsTool.invocationCount == 0, "Holdings tool should not have been called")
            #expect(transactionsTool.invocationCount == 0, "Transactions tool should not have been called")
            #expect(portfolioTool.invocationCount > 0, "Portfolio tool should have been called")
            #expect(!response.isEmpty, "Response should not be empty")
        }
        
        @Test
        func portfolioMultipleToolsCall() async throws {
            let (holdingsTool, transactionsTool, portfolioTool) = Self.createMockPortfolioTools()
            
            let session = LLMSession(
                model: Self.makeToolModel(),
                messages: [.system(createSystemPrompt())],
                tools: [holdingsTool, transactionsTool, portfolioTool]
            )
            
            let response = try await session.respond(to: "Give me a portfolio summary")
            print("Portfolio summary response: \(response)")
            
            #expect(holdingsTool.invocationCount > 0, "Holdings tool should have been called")
            #expect(transactionsTool.invocationCount > 0, "Transactions tool should have been called")
            #expect(portfolioTool.invocationCount > 0, "Portfolio tool should have been called")
            #expect(!response.isEmpty, "Response should not be empty")
            #expect(response.count > 100, "Summary should be substantial")
        }
        
        @Test
        func portfolioFilteredQuery() async throws {
            let (holdingsTool, transactionsTool, portfolioTool) = Self.createMockPortfolioTools()
            
            let session = LLMSession(
                model: Self.makeToolModel(),
                messages: [.system(createSystemPrompt())],
                tools: [holdingsTool, transactionsTool, portfolioTool]
            )
            
            let response = try await session.respond(to: "What percent of my portfolio is in fixed income?")
            print("Filtered query response: \(response)")
            
            #expect(holdingsTool.invocationCount > 0, "Holdings tool should have been called for percentage calculation")
            #expect(!response.isEmpty, "Response should not be empty")
        }
        
        @Test
        func portfolioDateInterpretation() async throws {
            let (holdingsTool, transactionsTool, portfolioTool) = Self.createMockPortfolioTools()
            
            let session = LLMSession(
                model: Self.makeToolModel(),
                messages: [.system(createSystemPrompt())],
                tools: [holdingsTool, transactionsTool, portfolioTool]
            )
            
            let response = try await session.respond(to: "What transactions did I have last August?")
            print("Date interpretation response: \(response)")
            
            #expect(transactionsTool.invocationCount > 0, "Transactions tool should have been called")
            #expect(!response.isEmpty, "Response should not be empty")
        }
        
        @Test
        func portfolioEmptyDataHandling() async throws {
            // Create tools with empty data directly
            let emptyHoldingsProvider: @Sendable () -> [Holding] = { [] }
            let emptyTransactionsProvider: @Sendable () -> [Transaction] = { [] }
            let emptyPortfolioProvider: @Sendable () -> [PortfolioValue] = { [] }
            
            let holdingsTool = GetHoldingsTool(provider: emptyHoldingsProvider)
            let transactionsTool = GetTransactionsTool(provider: emptyTransactionsProvider)
            let portfolioTool = GetPortfolioValTool(provider: emptyPortfolioProvider)
            
            let session = LLMSession(
                model: Self.makeToolModel(),
                messages: [.system(createSystemPrompt())],
                tools: [holdingsTool, transactionsTool, portfolioTool]
            )
            
            let response = try await session.respond(to: "Show me my portfolio summary")
            print("Empty data response: \(response)")
            
            #expect(!response.isEmpty, "Should handle empty data gracefully")
        }
    }
    
    // MARK: - Llama Model Comparison Tests
    
    @Suite(.serialized, .timeLimit(.minutes(30)))
    struct LlamaModelComparisonTests {
        
        private func runLlamaPortfolioQuery(
            _ query: String,
            model: LLMModel
        ) async throws -> (
            output: String,
            holdingsTool: GetHoldingsTool,
            transactionsTool: GetTransactionsTool,
            portfolioTool: GetPortfolioValTool
        ) {
            let startTime = Date()
            
            let data = Data(mockData.utf8)
            let container = try! JSONDecoder().decode(MockDataContainer.self, from: data)
            let holdingsTool = GetHoldingsTool(provider: { container.holdings })
            let transactionsTool = GetTransactionsTool(provider: { container.transactions })
            let portfolioTool = GetPortfolioValTool(provider: { container.portfolio_value })

            let session = LLMSession(
                model: LLMSessionLlamaTests.makeToolModel(model: model),
                messages: [.system(createSystemPrompt())],
                tools: [holdingsTool, transactionsTool, portfolioTool]
            )
            
            let output = try await session.respond(to: query)
            let elapsedTime = Date().timeIntervalSince(startTime)
            
            print("🔍 \(model.name) responded to '\(query)' in \(String(format: "%.2f", elapsedTime))s")
            
            return (output, holdingsTool, transactionsTool, portfolioTool)
        }
        
        @Test
        func testAllLlamaModels() async throws {
            let llamaModels: [LLMModel] = [.phi4mini, .gemma3_4b]
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

            var results: [String] = []
            
            for model in llamaModels {
                results.append("\nTesting \(model.name)")
                var modelPassed = 0
                var modelTotal = 0
                
                for (query, expectHoldings, expectTransactions, expectPortfolio) in allTestCases {
                    do {
                        let startTime = Date()
                        let (output, h, t, p) = try await runLlamaPortfolioQuery(query, model: model)
                        let elapsedTime = Date().timeIntervalSince(startTime)
                        
                        // Validate tool calls
                        var passed = true
                        var validationErrors: [String] = []
                        
                        if expectHoldings && h.invocationCount == 0 {
                            passed = false
                            validationErrors.append("Expected holdings tool to be called")
                        }
                        if !expectHoldings && h.invocationCount > 0 {
                            passed = false
                            validationErrors.append("Expected holdings tool NOT to be called")
                        }
                        if expectTransactions && t.invocationCount == 0 {
                            passed = false
                            validationErrors.append("Expected transactions tool to be called")
                        }
                        if !expectTransactions && t.invocationCount > 0 {
                            passed = false
                            validationErrors.append("Expected transactions tool NOT to be called")
                        }
                        if expectPortfolio && p.invocationCount == 0 {
                            passed = false
                            validationErrors.append("Expected portfolio tool to be called")
                        }
                        if !expectPortfolio && p.invocationCount > 0 {
                            passed = false
                            validationErrors.append("Expected portfolio tool NOT to be called")
                        }
                        if output.isEmpty {
                            passed = false
                            validationErrors.append("Expected non-empty output")
                        }
                        
                        modelTotal += 1
                        if passed { modelPassed += 1 }
                        
                        let status = passed ? "Yes" : "No"
                        results.append("  \(status) \(query.prefix(40))... [\(String(format: "%.2f", elapsedTime))s] Tools: H:\(h.invocationCount) T:\(t.invocationCount) P:\(p.invocationCount)")
                        
                        if !validationErrors.isEmpty {
                            results.append("    Errors: \(validationErrors.joined(separator: "; "))")
                        }
                        
                        // Still assert for test framework
                        if expectHoldings { #expect(h.invocationCount > 0) } else { #expect(h.invocationCount == 0) }
                        if expectTransactions { #expect(t.invocationCount > 0) } else { #expect(t.invocationCount == 0) }
                        if expectPortfolio { #expect(p.invocationCount > 0) } else { #expect(p.invocationCount == 0) }
                        #expect(!output.isEmpty)
                        
                    } catch {
                        modelTotal += 1
                        results.append("\(query.prefix(40))... ERROR: \(error.localizedDescription)")
                        throw error
                    }
                }
                
                results.append("\(model.name) Results: \(modelPassed)/\(modelTotal) passed (\(String(format: "%.1f", Double(modelPassed)/Double(modelTotal)*100))%)")
            }
            
            // Print final comparison report
            print("\n" + String(repeating: "=", count: 60))
            print("LLAMA MODEL COMPARISON RESULTS")
            print(String(repeating: "=", count: 60))
            for result in results {
                print(result)
            }
            print(String(repeating: "=", count: 60))
        }
    }
}
