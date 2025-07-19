//
//  LLMModel.swift
//  LocalLLMClient
//
//  Created by Rosemary Yang on 7/20/25.
//


// Sources/LocalLLMClient/LLMModel.swift

public enum LLMModel: Sendable, CaseIterable, Identifiable {
    case qwen3
    case qwen3_4b
    case qwen2_5VL_3b
    case gemma3_4b_mlx
    case phi4mini
    case gemma3
    case gemma3_4b
    case mobileVLM_3b

    public var name: String {
        switch self {
        case .qwen3: "MLX / Qwen3 1.7B"
        case .qwen3_4b: "MLX / Qwen3 4B"
        case .qwen2_5VL_3b: "MLX / Qwen2.5VL 3B"
        case .gemma3_4b_mlx: "MLX / Gemma3 4B"
        case .phi4mini: "llama.cpp / Phi-4 Mini 3.8B"
        case .gemma3: "llama.cpp / Gemma3 1B"
        case .gemma3_4b: "llama.cpp / Gemma3 4B"
        case .mobileVLM_3b: "llama.cpp / MobileVLM 3B"
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
        }
    }
    public var filename: String? {
        switch self {
        case .qwen3, .qwen3_4b, .qwen2_5VL_3b, .gemma3_4b_mlx: nil
        case .phi4mini: "Phi-4-mini-instruct-Q4_K_M.gguf"
        case .gemma3: "gemma-3-1B-it-QAT-Q4_0.gguf"
        case .gemma3_4b: "gemma-3-4B-it-QAT-Q4_0.gguf"
        case .mobileVLM_3b: "ggml-MobileVLM-3B-q5_k_s.gguf"
        }
    }
    public var mmprojFilename: String? {
        switch self {
        case .qwen3, .qwen3_4b, .qwen2_5VL_3b, .gemma3_4b_mlx, .phi4mini, .gemma3: nil
        #if os(macOS)
        case .gemma3_4b: "mmproj-model-f16.gguf"
        #elseif os(iOS)
        case .gemma3_4b: nil
        #endif
        case .mobileVLM_3b: "mmproj-model-f16.gguf"
        }
    }
    public var isMLX: Bool {
        filename == nil
    }
    public var supportsVision: Bool {
        switch self {
        case .qwen3, .qwen3_4b, .phi4mini, .gemma3: false
        #if os(macOS)
        case .gemma3_4b: true
        #elseif os(iOS)
        case .gemma3_4b: false
        #endif
        case .qwen2_5VL_3b, .gemma3_4b_mlx, .mobileVLM_3b: true
        }
    }
    public var extraEOSTokens: Set<String> {
        switch self {
        case .gemma3_4b_mlx:
            return ["<end_of_turn>"]
        case .qwen3, .qwen3_4b, .qwen2_5VL_3b, .phi4mini, .gemma3, .gemma3_4b, .mobileVLM_3b:
            return []
        }
    }
    public var supportsTools: Bool {
        switch self {
        case .qwen3, .qwen3_4b, .phi4mini, .gemma3, .gemma3_4b:
            return true
        case .qwen2_5VL_3b, .gemma3_4b_mlx, .mobileVLM_3b:
            return false
        }
    }
}
