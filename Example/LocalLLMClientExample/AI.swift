import Foundation
import LocalLLMClientCore
import LocalLLMClient
import LocalLLMClientMLX
import LocalLLMClientLlama
import LocalLLMClientFoundationModels
#if canImport(UIKit)
import UIKit
#endif

public enum LLMModel: Sendable, CaseIterable, Identifiable {
    case qwen3, qwen3_4b, qwen2_5VL_3b, gemma3_4b_mlx, phi4mini, gemma3, gemma3_4b, mobileVLM_3b, foundationModels

    var name: String {
        switch self {
        case .qwen3: "MLX / Qwen3 1.7B"
        case .qwen3_4b: "MLX / Qwen3 4B"
        case .qwen2_5VL_3b: "MLX / Qwen2.5VL 3B"
        case .gemma3_4b_mlx: "MLX / Gemma3 4B"
        case .phi4mini: "llama.cpp / Phi-4 Mini 3.8B"
        case .gemma3: "llama.cpp / Gemma3 1B"
        case .gemma3_4b: "llama.cpp / Gemma3 4B"
        case .mobileVLM_3b: "llama.cpp / MobileVLM 3B"
        case .foundationModels: "Apple Foundation Models"
        }
    }
    public var id: String {
        switch self {
        case .qwen3: "mlx-community/Qwen3-1.7B-4bit"
        case .qwen3_4b: "mlx-community/Qwen3-4B-4bit"
        case .qwen2_5VL_3b: "mlx-community/Qwen2.5-VL-3B-Instruct-abliterated-4bit"
        case .gemma3_4b_mlx: "mlx-community/gemma-3-4b-it-qat-4bit"
        case .phi4mini: "unsloth/Phi-4-mini-instruct-GGUF"
        case .gemma3: "lmstudio-community/gemma-3-1B-it-qat-GGUF"
        case .gemma3_4b: "lmstudio-community/gemma-3-4B-it-qat-GGUF"
        case .mobileVLM_3b: "Blombert/MobileVLM-3B-GGUF"
        case .foundationModels: "foundation-models"
        }
    }
    var filename: String? {
        switch self {
        case .qwen3, .qwen3_4b, .qwen2_5VL_3b, .gemma3_4b_mlx, .foundationModels: nil
        case .phi4mini: "Phi-4-mini-instruct-Q4_K_M.gguf"
        case .gemma3: "gemma-3-1B-it-QAT-Q4_0.gguf"
        case .gemma3_4b: "gemma-3-4B-it-QAT-Q4_0.gguf"
        case .mobileVLM_3b: "ggml-MobileVLM-3B-q5_k_s.gguf"
        }
    }
    var mmprojFilename: String? {
        switch self {
        case .qwen3, .qwen3_4b, .qwen2_5VL_3b, .gemma3_4b_mlx, .phi4mini, .gemma3, .foundationModels: nil
        #if os(macOS)
        case .gemma3_4b: "mmproj-model-f16.gguf"
        #elseif os(iOS)
        case .gemma3_4b: nil
        #endif
        case .mobileVLM_3b: "mmproj-model-f16.gguf"
        }
    }
    var supportsVision: Bool {
        switch self {
        case .qwen3, .qwen3_4b, .phi4mini, .gemma3, .foundationModels: false
        #if os(macOS)
        case .gemma3_4b: true
        #elseif os(iOS)
        case .gemma3_4b: false
        #endif
        case .qwen2_5VL_3b, .gemma3_4b_mlx, .mobileVLM_3b: true
        }
    }
    var extraEOSTokens: Set<String> {
        switch self {
        case .gemma3_4b_mlx: return ["<end_of_turn>"]
        default: return []
        }
    }
    var supportsTools: Bool {
        switch self {
        case .qwen3, .qwen3_4b, .phi4mini, .gemma3, .gemma3_4b, .foundationModels: true
        case .qwen2_5VL_3b, .gemma3_4b_mlx, .mobileVLM_3b: false
        }
    }
}

@Observable @MainActor
final class AI {
    let tools: [any LLMTool]
    var model = LLMModel.qwen3 {
        didSet {
            areToolsEnabled = model.supportsTools && areToolsEnabled
        }
    }
    private(set) var isLoading = false
    private(set) var downloadProgress: Double = 0
    var areToolsEnabled = false
    private var session: LLMSession?

    init(mockData: String) {
        let container = loadMockDataContainer(from: mockData) ?? MockDataContainer()
        self.tools = makeLLMTools(container: container)
    }

    var messages: [LLMInput.Message] {
        get { session?.messages ?? [] }
        set { session?.messages = newValue }
    }

    func resetMessages() {
        messages = [.system("\(sysPrompt)")]
    }

    // MARK: - Model selection and session creation

    private func createSessionModel() -> any LLMSession.Model {
        switch model {
        case .foundationModels:
            if #available(macOS 26.0, iOS 26.0, *) {
                return FoundationSessionModel()
            } else {
                fatalError("FoundationModels only available on macOS 15/iOS 18+")
            }
        case .qwen3, .qwen3_4b, .qwen2_5VL_3b, .gemma3_4b_mlx:
            return LLMSession.DownloadModel.mlx(
                id: model.id,
                parameter: .init(options: .init(extraEOSTokens: model.extraEOSTokens))
            )
        case .phi4mini, .gemma3, .gemma3_4b, .mobileVLM_3b:
            return LLMSession.DownloadModel.llama(
                id: model.id,
                model: model.filename!,
                mmproj: model.mmprojFilename,
                parameter: .init(context: 10240, options: .init(extraEOSTokens: model.extraEOSTokens, verbose: true))
            )
        }
    }

    func loadLLM() async {
        isLoading = true
        defer { isLoading = false }
        session = nil
        do {
            let sessionModel = createSessionModel()
            let toolsToUse: [any LLMTool] = areToolsEnabled ? tools : []
            // Download progress for models that support it
            if let downloadModel = sessionModel as? LLMSession.DownloadModel {
                try await downloadModel.downloadModel { @MainActor [weak self] progress in
                    self?.downloadProgress = progress
                    print("Download progress: \(progress)")
                }
            }
            session = LLMSession(model: sessionModel, tools: toolsToUse)
            resetMessages()
        } catch {
            print("Failed to load LLM: \(error)")
        }
    }

    func ask(_ message: String, attachments: [LLMAttachment]) async throws -> AsyncThrowingStream<String, any Error> {
        guard let session else {
            throw LLMError.failedToLoad(reason: "LLM not loaded")
        }
        return session.streamResponse(to: message, attachments: attachments)
    }

    func toggleTools() async {
        areToolsEnabled.toggle()
        if session != nil {
            await loadLLM()
        }
    }
}

// MARK: - Foundation Models Support

@available(macOS 26.0, iOS 26.0, *)
private struct FoundationSessionModel: LLMSession.Model {
    func prewarm() async throws {}
    let makeClient: @Sendable ([AnyLLMTool]) async throws -> AnyLLMClient =
        { (tools: [AnyLLMTool]) async throws -> AnyLLMClient in
            let asAnyLLMTools: [any LLMTool] = tools.map { $0.underlyingTool }
            FoundationModelsClient(
                model: .default,
                generationOptions: .init(),
                tools: asAnyLLMTools 
            )
        }
}

#if DEBUG
extension AI {
    func setSession(_ session: LLMSession) {
        self.session = session
    }
}
#endif
