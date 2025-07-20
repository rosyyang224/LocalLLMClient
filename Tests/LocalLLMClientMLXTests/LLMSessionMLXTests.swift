import Testing
import Foundation
import LocalLLMClient
import LocalLLMClientCore
import LocalLLMClientMLX
import LocalLLMClientTestUtilities

// MARK: - Test Result Tracking

struct TestResult {
    let modelName: String
    let modelId: String
    let query: String
    let output: String
    let holdingsToolCalls: Int
    let transactionsToolCalls: Int
    let portfolioToolCalls: Int
    let elapsedTime: TimeInterval
    let passed: Bool
    let errorMessage: String?
    
    // Computed metrics
    var outputWordCount: Int { output.split(separator: " ").count }
    var outputCharacterCount: Int { output.count }
    var totalToolCalls: Int { holdingsToolCalls + transactionsToolCalls + portfolioToolCalls }
    var outputLinesCount: Int { output.components(separatedBy: .newlines).count }
    
    // Quality metrics (simple heuristics)
    var hasNumbers: Bool { output.rangeOfCharacter(from: .decimalDigits) != nil }
    var hasCurrency: Bool { output.contains("$") || output.lowercased().contains("dollar") }
    var mentionsPortfolio: Bool { output.lowercased().contains("portfolio") }
    var mentionsHoldings: Bool { output.lowercased().contains("holding") }
    var mentionsTransactions: Bool { output.lowercased().contains("transaction") }
    
    var concisenesScore: Double {
        // Lower word count for same information = better conciseness
        // This is a simple heuristic - could be improved
        let targetWordCount = 100.0
        let wordCount = Double(outputWordCount)
        return max(0, min(1, targetWordCount / wordCount))
    }
    
    var relevanceScore: Double {
        var score = 0.0
        let queryLower = query.lowercased()
        
        // Check if output mentions relevant topics based on query
        if queryLower.contains("holding") && mentionsHoldings { score += 0.3 }
        if queryLower.contains("transaction") && mentionsTransactions { score += 0.3 }
        if queryLower.contains("portfolio") && mentionsPortfolio { score += 0.3 }
        if queryLower.contains("summary") && (mentionsPortfolio || mentionsHoldings) { score += 0.2 }
        if queryLower.contains("overview") && (mentionsPortfolio || mentionsHoldings) { score += 0.2 }
        if (queryLower.contains("value") || queryLower.contains("$")) && hasCurrency { score += 0.2 }
        
        return min(1.0, score)
    }
}

@MainActor
class TestResultCollector: @unchecked Sendable {
    private var results: [TestResult] = []
    private let startTime = Date()
    
    func addResult(_ result: TestResult) {
        results.append(result)
    }
    
    func generateReport() -> String {
        var report = """
        
        ================== TEST RESULTS REPORT ==================
        Total Tests Run: \(results.count)
        Passed: \(results.filter { $0.passed }.count)
        Failed: \(results.filter { !$0.passed }.count)
        Total Runtime: \(String(format: "%.2f", Date().timeIntervalSince(startTime))) seconds
        
        """
        
        // Group by model
        let modelGroups = Dictionary(grouping: results) { $0.modelName }
        
        for (modelName, modelResults) in modelGroups.sorted(by: { $0.key < $1.key }) {
            let passed = modelResults.filter { $0.passed }.count
            let total = modelResults.count
            let avgTime = modelResults.map { $0.elapsedTime }.reduce(0, +) / Double(modelResults.count)
            let avgWordCount = modelResults.map { Double($0.outputWordCount) }.reduce(0, +) / Double(modelResults.count)
            let avgRelevance = modelResults.map { $0.relevanceScore }.reduce(0, +) / Double(modelResults.count)
            let avgConciseness = modelResults.map { $0.concisenesScore }.reduce(0, +) / Double(modelResults.count)
            
            report += """
            
            --- \(modelName) ---
            Success Rate: \(passed)/\(total) (\(String(format: "%.1f", Double(passed)/Double(total)*100))%)
            Avg Response Time: \(String(format: "%.2f", avgTime))s
            Avg Word Count: \(String(format: "%.1f", avgWordCount))
            Avg Relevance Score: \(String(format: "%.2f", avgRelevance))
            Avg Conciseness Score: \(String(format: "%.2f", avgConciseness))
            
            """
            
            // Show individual test details
            for result in modelResults {
                let status = result.passed ? "✅" : "❌"
                report += """
                  \(status) \(result.query.prefix(50))... 
                     Tools: H:\(result.holdingsToolCalls) T:\(result.transactionsToolCalls) P:\(result.portfolioToolCalls) | Time: \(String(format: "%.2f", result.elapsedTime))s | Words: \(result.outputWordCount)
                """
                if let error = result.errorMessage {
                    report += " | Error: \(error)"
                }
                report += "\n"
            }
        }
        
        // Best performers
        let successfulResults = results.filter { $0.passed }
        if !successfulResults.isEmpty {
            report += "\n--- TOP PERFORMERS ---\n"
            
            let fastestResults = successfulResults.sorted { $0.elapsedTime < $1.elapsedTime }.prefix(3)
            report += "Fastest Responses:\n"
            for result in fastestResults {
                report += "  \(result.modelName): \(String(format: "%.2f", result.elapsedTime))s - \(result.query.prefix(30))...\n"
            }
            
            let mostRelevant = successfulResults.sorted { $0.relevanceScore > $1.relevanceScore }.prefix(3)
            report += "\nMost Relevant Responses:\n"
            for result in mostRelevant {
                report += "  \(result.modelName): \(String(format: "%.2f", result.relevanceScore)) - \(result.query.prefix(30))...\n"
            }
            
            let mostConcise = successfulResults.sorted { $0.concisenesScore > $1.concisenesScore }.prefix(3)
            report += "\nMost Concise Responses:\n"
            for result in mostConcise {
                report += "  \(result.modelName): \(String(format: "%.2f", result.concisenesScore)) - \(result.query.prefix(30))...\n"
            }
        }
        
        report += "\n========================================================\n"
        return report
    }
    
    func exportCSV() -> String {
        var csv = "Model Name,Model ID,Query,Passed,Elapsed Time,Output Word Count,Output Char Count,Total Tool Calls,Holdings Calls,Transactions Calls,Portfolio Calls,Has Numbers,Has Currency,Relevance Score,Conciseness Score,Error Message\n"
        
        for result in results {
            csv += "\"\(result.modelName)\",\"\(result.modelId)\",\"\(result.query.replacingOccurrences(of: "\"", with: "\"\""))\",\(result.passed),\(result.elapsedTime),\(result.outputWordCount),\(result.outputCharacterCount),\(result.totalToolCalls),\(result.holdingsToolCalls),\(result.transactionsToolCalls),\(result.portfolioToolCalls),\(result.hasNumbers),\(result.hasCurrency),\(result.relevanceScore),\(result.concisenesScore),\"\(result.errorMessage?.replacingOccurrences(of: "\"", with: "\"\"") ?? "")\"\n"
        }
        
        return csv
    }
}

// Global result collector
@MainActor
let testResultCollector = TestResultCollector()

func makeDownloadModel(model: LLMModel) -> LLMSession.DownloadModel {
    .mlx(id: model.id)
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
    let startTime = Date()
    
    let data = Data(mockData.utf8)
    let container = try! JSONDecoder().decode(MockDataContainer.self, from: data)
    let holdingsTool = GetHoldingsTool(provider: { container.holdings })
    let transactionsTool = GetTransactionsTool(provider: { container.transactions })
    let portfolioTool = GetPortfolioValTool(provider: { container.portfolio_value })

    let session = LLMSession(
        model: makeDownloadModel(model: model),
        messages: [.system(sysPrompt)],
        tools: [holdingsTool, transactionsTool, portfolioTool]
    )
    
    let output = try await session.respond(to: query)
    let elapsedTime = Date().timeIntervalSince(startTime)
    
    // Record the result
    await MainActor.run {
        let result = TestResult(
            modelName: model.name,
            modelId: model.id,
            query: query,
            output: output,
            holdingsToolCalls: holdingsTool.invocationCount,
            transactionsToolCalls: transactionsTool.invocationCount,
            portfolioToolCalls: portfolioTool.invocationCount,
            elapsedTime: elapsedTime,
            passed: true, // Will be updated if validation fails
            errorMessage: nil
        )
        testResultCollector.addResult(result)
    }
    
    return (output, holdingsTool, transactionsTool, portfolioTool)
}

extension ModelTests {
    @Suite
    struct DebugTests {
        @Test(.timeLimit(.minutes(1)))
        func testBasicSetup() async throws {
            // Test that our enum works
            let model = LLMModel.qwen3
            
            // Test mock data loading
            let data = Data(mockData.utf8)
            let container = try JSONDecoder().decode(MockDataContainer.self, from: data)
            
            // Test tool creation
            let holdingsTool = GetHoldingsTool(provider: { container.holdings })
            let transactionsTool = GetTransactionsTool(provider: { container.transactions })
            let portfolioTool = GetPortfolioValTool(provider: { container.portfolio_value })
        }
        
        @Test(.timeLimit(.minutes(1)))
        func testModelCreation() async throws {
            let model = LLMModel.qwen3
            let downloadModel = makeDownloadModel(model: model)
        }
    }

    @Suite
    struct HoldingsToolTests {
        @Test(.timeLimit(.minutes(1))) func testUSEquityHoldings() async throws {
            let (output, h, t, p) = try await runPortfolioQuery("What are my US equity holdings?")
            #expect(h.invocationCount > 0)
            #expect(t.invocationCount == 0)
            #expect(p.invocationCount == 0)
            #expect(!output.isEmpty)
        }
        @Test(.timeLimit(.minutes(1))) func testFixedIncomeHoldings() async throws {
            let (output, h, t, p) = try await runPortfolioQuery("List all fixed income holdings.")
            #expect(h.invocationCount > 0)
            #expect(t.invocationCount == 0)
            #expect(p.invocationCount == 0)
            #expect(!output.isEmpty)
        }
        @Test(.timeLimit(.minutes(1))) func testHoldingsByRegion() async throws {
            let (output, h, t, p) = try await runPortfolioQuery("Show my holdings in the United States.")
            #expect(h.invocationCount > 0)
            #expect(t.invocationCount == 0)
            #expect(p.invocationCount == 0)
            #expect(!output.isEmpty)
        }
    }

    @Suite
    struct PortfolioToolTests {
        @Test(.timeLimit(.minutes(1))) func testTotalMarketValue() async throws {
            let (output, h, t, p) = try await runPortfolioQuery("Calculate total market value across all accounts.")
            #expect(h.invocationCount == 0)
            #expect(t.invocationCount == 0)
            #expect(p.invocationCount > 0)
            #expect(!output.isEmpty)
        }
        @Test(.timeLimit(.minutes(1))) func testPortfolioTrend() async throws {
            let (output, h, t, p) = try await runPortfolioQuery("Show me the trend in my portfolio value.")
            #expect(h.invocationCount == 0)
            #expect(t.invocationCount == 0)
            #expect(p.invocationCount > 0)
            #expect(!output.isEmpty)
        }
        @Test(.timeLimit(.minutes(1))) func testHighestPortfolioValue() async throws {
            let (output, h, t, p) = try await runPortfolioQuery("What was my highest portfolio value?")
            #expect(h.invocationCount == 0)
            #expect(t.invocationCount == 0)
            #expect(p.invocationCount > 0)
            #expect(!output.isEmpty)
        }
    }
    
    @Suite
    struct TransactionToolTests {
        @Test(.timeLimit(.minutes(1))) func testRecentTransactions() async throws {
            let (output, h, t, p) = try await runPortfolioQuery("What are my most recent transactions?")
            #expect(h.invocationCount == 0)
            #expect(t.invocationCount > 0)
            #expect(p.invocationCount == 0)
            #expect(!output.isEmpty)
        }
        @Test(.timeLimit(.minutes(1))) func testTransactionsByType() async throws {
            let (output, h, t, p) = try await runPortfolioQuery("Show me all BUY transactions.")
            #expect(h.invocationCount == 0)
            #expect(t.invocationCount > 0)
            #expect(p.invocationCount == 0)
            #expect(!output.isEmpty)
        }
        @Test(.timeLimit(.minutes(1))) func testLargeTransactions() async throws {
            let (output, h, t, p) = try await runPortfolioQuery("List all transactions greater than $10,000.")
            #expect(h.invocationCount == 0)
            #expect(t.invocationCount > 0)
            #expect(p.invocationCount == 0)
            #expect(!output.isEmpty)
        }
    }
    
    @Suite
    struct CrossDataToolTests {
        @Test(.timeLimit(.minutes(1))) func testMultiAspectQuery() async throws {
            let (output, h, t, p) = try await runPortfolioQuery("Give me an overview of my holdings and recent transactions in the last month.")
            #expect(h.invocationCount > 0)
            #expect(t.invocationCount > 0)
            #expect(p.invocationCount == 0)
            #expect(!output.isEmpty)
        }
        @Test(.timeLimit(.minutes(1))) func testPercentFixedIncome() async throws {
            let (output, h, t, p) = try await runPortfolioQuery("What percent of my portfolio is in fixed income?")
            #expect(h.invocationCount > 0)
            #expect(t.invocationCount == 0)
            #expect(p.invocationCount == 0)
            #expect(!output.isEmpty)
        }
        @Test(.timeLimit(.minutes(1))) func testPerformanceByAssetClass() async throws {
            let (output, h, t, p) = try await runPortfolioQuery("Compare equity vs fixed income returns.")
            #expect(h.invocationCount > 0)
            #expect(t.invocationCount == 0)
            #expect(p.invocationCount == 0)
            #expect(!output.isEmpty)
        }
        @Test(.timeLimit(.minutes(1))) func testBreakdownIntlVsUS() async throws {
            let (output, h, t, p) = try await runPortfolioQuery("Break down my international vs US investments.")
            #expect(h.invocationCount > 0)
            #expect(t.invocationCount == 0)
            #expect(p.invocationCount == 0)
            #expect(!output.isEmpty)
        }
    }
    
    @Suite
    struct SummarizationTests {
        @Test(.timeLimit(.minutes(1))) func testPortfolioSummary() async throws {
            let (output, h, t, p) = try await runPortfolioQuery("Show me a summary of my portfolio")
            #expect(h.invocationCount > 0)
            #expect(t.invocationCount > 0)
            #expect(p.invocationCount > 0)
            #expect(!output.isEmpty)
        }
        @Test(.timeLimit(.minutes(1))) func testFullPortfolioOverview() async throws {
            let (output, h, t, p) = try await runPortfolioQuery("Give me a full portfolio overview.")
            #expect(h.invocationCount > 0)
            #expect(t.invocationCount > 0)
            #expect(p.invocationCount > 0)
            #expect(!output.isEmpty)
        }
    }
    
    @Suite(.serialized, .timeLimit(.minutes(30)))
    struct AllToolTestSweep {
        @Test
        func testAllMLXModels() async throws {
            let allMLXModels: [LLMModel] = LLMModel.allCases.filter { $0.isMLX && $0.supportsTools }
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

            for model in allMLXModels {
                for (query, expectHoldings, expectTransactions, expectPortfolio) in allTestCases {
                    do {
                        let startTime = Date()
                        let (output, h, t, p) = try await runPortfolioQuery(query, model: model)
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
                        
                        // Update the result with validation outcome
                        await MainActor.run {
                            let result = TestResult(
                                modelName: model.name,
                                modelId: model.id,
                                query: query,
                                output: output,
                                holdingsToolCalls: h.invocationCount,
                                transactionsToolCalls: t.invocationCount,
                                portfolioToolCalls: p.invocationCount,
                                elapsedTime: elapsedTime,
                                passed: passed,
                                errorMessage: validationErrors.isEmpty ? nil : validationErrors.joined(separator: "; ")
                            )
                            testResultCollector.addResult(result)
                        }
                        
                        // Still assert for test framework
                        if expectHoldings { #expect(h.invocationCount > 0) } else { #expect(h.invocationCount == 0) }
                        if expectTransactions { #expect(t.invocationCount > 0) } else { #expect(t.invocationCount == 0) }
                        if expectPortfolio { #expect(p.invocationCount > 0) } else { #expect(p.invocationCount == 0) }
                        #expect(!output.isEmpty)
                        
                    } catch {
                        // Record failed test
                        await MainActor.run {
                            let result = TestResult(
                                modelName: model.name,
                                modelId: model.id,
                                query: query,
                                output: "",
                                holdingsToolCalls: 0,
                                transactionsToolCalls: 0,
                                portfolioToolCalls: 0,
                                elapsedTime: 0,
                                passed: false,
                                errorMessage: error.localizedDescription
                            )
                            testResultCollector.addResult(result)
                        }
                        throw error
                    }
                }
            }
            
            // Print final report
            await MainActor.run {
                print(testResultCollector.generateReport())
                print("\n--- CSV Export ---")
                print(testResultCollector.exportCSV())
            }
        }
    }
}
