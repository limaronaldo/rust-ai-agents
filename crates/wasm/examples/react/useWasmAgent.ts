/**
 * React hook for using Rust AI Agents in the browser.
 *
 * @example
 * ```tsx
 * import { useWasmAgent } from './useWasmAgent';
 *
 * function ChatComponent() {
 *   const { sendMessage, messages, isLoading, error } = useWasmAgent({
 *     provider: 'openai',
 *     apiKey: process.env.OPENAI_API_KEY!,
 *     model: 'gpt-4o-mini',
 *     systemPrompt: 'You are a helpful assistant.',
 *   });
 *
 *   return (
 *     <div>
 *       {messages.map((msg, i) => (
 *         <div key={i} className={msg.role}>
 *           {msg.content}
 *         </div>
 *       ))}
 *       <button onClick={() => sendMessage('Hello!')}>
 *         Send
 *       </button>
 *     </div>
 *   );
 * }
 * ```
 */

import { useState, useEffect, useCallback, useRef } from 'react';

// Types for the WASM module
interface WasmAgentConfig {
  provider: string;
  apiKey: string;
  model: string;
  systemPrompt?: string;
  temperature?: number;
  maxTokens?: number;
}

interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

interface UseWasmAgentOptions extends WasmAgentConfig {
  onError?: (error: Error) => void;
  onMessage?: (message: Message) => void;
}

interface UseWasmAgentReturn {
  sendMessage: (content: string) => Promise<void>;
  sendMessageStream: (content: string) => Promise<void>;
  messages: Message[];
  isLoading: boolean;
  isStreaming: boolean;
  error: Error | null;
  clearHistory: () => void;
  isReady: boolean;
}

// Dynamic import of WASM module
let wasmModule: any = null;
let wasmInitPromise: Promise<any> | null = null;

async function loadWasm() {
  if (wasmModule) return wasmModule;
  if (wasmInitPromise) return wasmInitPromise;

  wasmInitPromise = (async () => {
    // Adjust path based on your setup
    const wasm = await import('rust-ai-agents-wasm');
    await wasm.default();
    wasmModule = wasm;
    return wasm;
  })();

  return wasmInitPromise;
}

export function useWasmAgent(options: UseWasmAgentOptions): UseWasmAgentReturn {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [isReady, setIsReady] = useState(false);

  const agentRef = useRef<any>(null);
  const optionsRef = useRef(options);
  optionsRef.current = options;

  // Initialize WASM and create agent
  useEffect(() => {
    let mounted = true;

    async function init() {
      try {
        const wasm = await loadWasm();

        if (!mounted) return;

        const { WasmAgent, WasmAgentConfig } = wasm;

        const config = new WasmAgentConfig(
          options.provider,
          options.apiKey,
          options.model
        );

        if (options.systemPrompt) {
          config.system_prompt = options.systemPrompt;
        }
        if (options.temperature !== undefined) {
          config.temperature = options.temperature;
        }
        if (options.maxTokens !== undefined) {
          config.max_tokens = options.maxTokens;
        }

        agentRef.current = new WasmAgent(config);
        setIsReady(true);
        setError(null);

      } catch (err) {
        if (mounted) {
          const error = err instanceof Error ? err : new Error(String(err));
          setError(error);
          options.onError?.(error);
        }
      }
    }

    init();

    return () => {
      mounted = false;
      if (agentRef.current) {
        agentRef.current.free();
        agentRef.current = null;
      }
    };
  }, [options.provider, options.apiKey, options.model]);

  // Send message (non-streaming)
  const sendMessage = useCallback(async (content: string) => {
    if (!agentRef.current || isLoading) return;

    const userMessage: Message = { role: 'user', content };
    setMessages(prev => [...prev, userMessage]);
    optionsRef.current.onMessage?.(userMessage);

    setIsLoading(true);
    setError(null);

    try {
      const response = await agentRef.current.chat(content);
      const assistantMessage: Message = {
        role: 'assistant',
        content: response.content
      };
      setMessages(prev => [...prev, assistantMessage]);
      optionsRef.current.onMessage?.(assistantMessage);

    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      setError(error);
      optionsRef.current.onError?.(error);

      const errorMessage: Message = {
        role: 'system',
        content: `Error: ${error.message}`
      };
      setMessages(prev => [...prev, errorMessage]);

    } finally {
      setIsLoading(false);
    }
  }, [isLoading]);

  // Send message (streaming)
  const sendMessageStream = useCallback(async (content: string) => {
    if (!agentRef.current || isStreaming) return;

    const userMessage: Message = { role: 'user', content };
    setMessages(prev => [...prev, userMessage]);
    optionsRef.current.onMessage?.(userMessage);

    setIsStreaming(true);
    setError(null);

    // Add placeholder for assistant response
    let fullResponse = '';
    setMessages(prev => [...prev, { role: 'assistant', content: '' }]);

    try {
      const stream = await agentRef.current.chatStream(content);
      const reader = stream.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });

        // Parse SSE data
        const lines = chunk.split('\n');
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') continue;

            try {
              const parsed = JSON.parse(data);
              const text = parsed.choices?.[0]?.delta?.content ||
                          parsed.delta?.text || '';

              if (text) {
                fullResponse += text;
                setMessages(prev => {
                  const updated = [...prev];
                  updated[updated.length - 1] = {
                    role: 'assistant',
                    content: fullResponse
                  };
                  return updated;
                });
              }
            } catch {
              // Not JSON
            }
          }
        }
      }

      const assistantMessage: Message = {
        role: 'assistant',
        content: fullResponse
      };
      optionsRef.current.onMessage?.(assistantMessage);

    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      setError(error);
      optionsRef.current.onError?.(error);

      setMessages(prev => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          role: 'system',
          content: `Error: ${error.message}`
        };
        return updated;
      });

    } finally {
      setIsStreaming(false);
    }
  }, [isStreaming]);

  // Clear history
  const clearHistory = useCallback(() => {
    if (agentRef.current) {
      agentRef.current.clear();
    }
    setMessages([]);
    setError(null);
  }, []);

  return {
    sendMessage,
    sendMessageStream,
    messages,
    isLoading,
    isStreaming,
    error,
    clearHistory,
    isReady,
  };
}

export default useWasmAgent;
