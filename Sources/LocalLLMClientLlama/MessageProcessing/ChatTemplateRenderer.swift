import LocalLLMClientCore
import Foundation
import Jinja

/// Context for template rendering
struct TemplateContext {
    let specialTokens: [String: String]
    let additionalContext: [String: Any]
    
    init(
        specialTokens: [String: String] = [:],
        additionalContext: [String: Any] = [:]
    ) {
        self.specialTokens = specialTokens
        self.additionalContext = additionalContext
    }
}

/// Protocol for rendering chat templates
 protocol ChatTemplateRenderer: Sendable {
    /// Render messages using a chat template
    func render(
        messages: [LLMInput.ChatTemplateMessage],
        template: String,
        context: TemplateContext,
        tools: [AnyLLMTool]
    ) throws(LLMError) -> String
}

/// Standard Jinja-based template renderer
 struct JinjaChatTemplateRenderer: ChatTemplateRenderer {
    private let toolProcessor: ToolInstructionProcessor
    
     init(toolProcessor: ToolInstructionProcessor = StandardToolInstructionProcessor()) {
        self.toolProcessor = toolProcessor
    }
    
     func render(
        messages: [LLMInput.ChatTemplateMessage],
        template: String,
        context: TemplateContext,
        tools: [AnyLLMTool]
    ) throws(LLMError) -> String {
        print("🎨 ChatTemplateRenderer.render called")
        print("🎨 Received \(tools.count) tools: \(tools.map { $0.name })")
//        print("🎨 Template preview: \(String(template.prefix(200)))...")
        let jinjaTemplate: Template
        do {
            jinjaTemplate = try Template(template)
        } catch {
            throw LLMError.invalidParameter(reason: "Failed to parse template: \(error.localizedDescription)")
        }
        
        // Extract message data
        var messagesData = messages.map(\.value)
        print("🎨 Original messages count: \(messagesData.count)")
//        print("🎨 Original messages: \(messagesData)")
        
        // Process tool instructions if needed
        let hasNativeToolSupport = toolProcessor.hasNativeToolSupport(in: template)
        messagesData = try toolProcessor.processMessages(
            messagesData,
            tools: tools,
            templateHasNativeSupport: hasNativeToolSupport
        )
        print("🎨 After tool processing - messages count: \(messagesData.count)")
        print("🎨 After tool processing - messages: \(messagesData)")
        
        // Build template context
        let templateContext = buildTemplateContext(
            messages: messagesData,
            tools: tools,
            hasNativeToolSupport: hasNativeToolSupport,
            context: context
        )
        
        // Render template
        do {
            let result = try jinjaTemplate.render(templateContext)
            print("🎨 Rendered result length: \(result.count)")
            print("🎨 Result contains '<tool_call>': \(result.contains("<tool_call>"))")
            print("🎨 Result contains 'get_holdings': \(result.contains("get_holdings"))")
            print("🎨 Result contains 'tool_call': \(result.contains("tool_call"))")
            return result
        } catch {
            throw LLMError.invalidParameter(reason: "Failed to render template: \(error.localizedDescription)")
        }
    }
    
    private func buildTemplateContext(
        messages: [[String: any Sendable]],
        tools: [AnyLLMTool],
        hasNativeToolSupport: Bool,
        context: TemplateContext
    ) -> [String: Any] {
        var templateContext: [String: Any] = [
            "add_generation_prompt": true,
            "messages": messages
        ]
        
        // Add special tokens
        templateContext.merge(context.specialTokens) { _, new in new }
        
        // Add tools for templates with native support
        if !tools.isEmpty && hasNativeToolSupport {
            templateContext["tools"] = tools.compactMap { $0.toOAICompatJSON() }
        }
        
        // Add additional context
        templateContext.merge(context.additionalContext) { _, new in new }
        
        return templateContext
    }
}
