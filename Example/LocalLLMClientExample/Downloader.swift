import SwiftUI
import LocalLLMClientUtility

struct Downloader: Sendable {
    init(model: LLMModel) {
        self.model = model
        // No download needed for FoundationModels!
        guard model != .foundationModels else {
            downloader = nil
            return
        }
        let globs: Globs = switch model {
        case .qwen3, .qwen3_4b, .qwen2_5VL_3b, .gemma3_4b_mlx:
            .mlx
        case .phi4mini, .gemma3, .gemma3_4b, .mobileVLM_3b:
            .init(
                (model.filename.map { [$0] } ?? []) + (model.mmprojFilename.map { [$0] } ?? [])
            )
        case .foundationModels:
            .mlx // doesn't matter, this path never runs
        }
    #if os(macOS)
        downloader = FileDownloader(source: .huggingFace(id: model.id, globs: globs))
    #elseif os(iOS)
        downloader = FileDownloader(
            source: .huggingFace(id: model.id, globs: globs),
            configuration: .background(withIdentifier: "localllmclient.downloader.\(model.id)")
        )
    #endif
        // try? downloader.removeMetadata() // use if you update the models
    }

    private let model: LLMModel
    private let downloader: FileDownloader?

    var url: URL? {
        guard let filename = model.filename, let downloader else { return nil }
        return downloader.destination.appending(component: filename)
    }

    var clipURL: URL? {
        guard let mmproj = model.mmprojFilename, let downloader else { return nil }
        return downloader.destination.appending(component: mmproj)
    }

    var isDownloaded: Bool {
        guard let downloader else { return true } // Always "downloaded" for FoundationModels
        return downloader.isDownloaded
    }

    func download(progressHandler: @escaping @Sendable (Double) async -> Void) async throws {
        guard let downloader else { return } // Skip for FoundationModels
        try await downloader.download { progress in
            await progressHandler(progress)
        }
    }
}
