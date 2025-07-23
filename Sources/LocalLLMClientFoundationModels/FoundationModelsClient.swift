#if canImport(FoundationModels)
import Foundation
import FoundationModels
import LocalLLMClient
import LocalLLMClientCore

@available(iOS 26.0, macOS 26.0, *)
@available(tvOS, unavailable)
@available(watchOS, unavailable)
public final actor FoundationModelsClient: LLMClient {
    let model: SystemLanguageModel
    let generationOptions: GenerationOptions
    let tools: [any Tool]

    public init(
        model: SystemLanguageModel,
        generationOptions: GenerationOptions,
        tools: [any Tool] = []
    ) {
        self.model = model
        self.generationOptions = generationOptions
        self.tools = tools
    }

    // MARK: - Core text output
    public func text(from input: LLMInput) async throws -> String {
        let session = try await LanguageModelSession(
            model: model,
            tools: tools,
            instructions: input.makePromptString()
        )
        let result = try await session.respond(to: input.makePromptString())
        return result.content
    }

    // MARK: - Streaming text output (no native streaming; yield all at once)
    public func textStream(from input: LLMInput) async throws -> AsyncStream<String> {
        let response = try await text(from: input)
        return AsyncStream { continuation in
            continuation.yield(response)
            continuation.finish()
        }
    }

    // MARK: - Response stream (just chunkless fallback)
    public func responseStream(from input: LLMInput) async throws -> AsyncThrowingStream<StreamingChunk, Error> {
        let response = try await text(from: input)
        return AsyncThrowingStream { continuation in
            continuation.yield(.text(response))
            continuation.finish()
        }
    }

    // MARK: - Tool calling
    public func generateToolCalls(from input: LLMInput) async throws -> LocalLLMClientCore.GeneratedContent {
        throw LLMError.invalidParameter(reason: "Tool calls are not supported by FoundationModelsClient in this SDK version")
    }

    public func resume(
        withToolCalls toolCalls: [LLMToolCall],
        toolOutputs: [(String, String)],
        originalInput: LLMInput
    ) async throws -> String {
        throw LLMError.invalidParameter(reason: "Tool calls are not supported by FoundationModelsClient in this SDK version")
    }
}

// MARK: - LLMInput Extensions for Prompt

@available(iOS 26.0, macOS 26.0, *)
@available(tvOS, unavailable)
@available(watchOS, unavailable)
extension LLMInput {
    func makePromptString() -> String {
        switch value {
        case let .plain(text):
            return text
        case let .chatTemplate(messages):
            return messages.last?.value["content"] as? String ?? ""
        case let .chat(messages):
            return messages.last?.content ?? ""
        }
    }
}
#endif
