/**
 * Example React component using the Rust AI Agents WASM module.
 *
 * @example
 * ```tsx
 * import { ChatComponent } from './ChatComponent';
 *
 * function App() {
 *   return (
 *     <ChatComponent
 *       provider="openai"
 *       apiKey={process.env.REACT_APP_OPENAI_API_KEY!}
 *       model="gpt-4o-mini"
 *     />
 *   );
 * }
 * ```
 */

import React, { useState, useRef, useEffect } from 'react';
import { useWasmAgent } from './useWasmAgent';

interface ChatComponentProps {
  provider: 'openai' | 'anthropic' | 'openrouter';
  apiKey: string;
  model: string;
  systemPrompt?: string;
  className?: string;
}

export function ChatComponent({
  provider,
  apiKey,
  model,
  systemPrompt,
  className = '',
}: ChatComponentProps) {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const {
    sendMessageStream,
    messages,
    isStreaming,
    error,
    clearHistory,
    isReady,
  } = useWasmAgent({
    provider,
    apiKey,
    model,
    systemPrompt,
    temperature: 0.7,
    maxTokens: 2048,
  });

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isStreaming) return;

    const message = input.trim();
    setInput('');
    await sendMessageStream(message);
  };

  return (
    <div className={`chat-component ${className}`} style={styles.container}>
      <div style={styles.header}>
        <h3 style={styles.title}>AI Chat</h3>
        <span style={styles.status}>
          {!isReady ? '🔄 Loading...' : isStreaming ? '💭 Thinking...' : '✅ Ready'}
        </span>
        <button onClick={clearHistory} style={styles.clearBtn}>
          Clear
        </button>
      </div>

      <div style={styles.messages}>
        {messages.length === 0 && (
          <div style={styles.emptyState}>
            Start a conversation...
          </div>
        )}

        {messages.map((msg, index) => (
          <div
            key={index}
            style={{
              ...styles.message,
              ...(msg.role === 'user' ? styles.userMessage : {}),
              ...(msg.role === 'assistant' ? styles.assistantMessage : {}),
              ...(msg.role === 'system' ? styles.systemMessage : {}),
            }}
          >
            <div style={styles.messageRole}>
              {msg.role === 'user' ? '👤 You' : msg.role === 'assistant' ? '🤖 AI' : '⚠️ System'}
            </div>
            <div style={styles.messageContent}>
              {msg.content || (isStreaming && index === messages.length - 1 ? '...' : '')}
            </div>
          </div>
        ))}

        <div ref={messagesEndRef} />
      </div>

      {error && (
        <div style={styles.error}>
          Error: {error.message}
        </div>
      )}

      <form onSubmit={handleSubmit} style={styles.inputArea}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={isReady ? "Type your message..." : "Loading..."}
          disabled={!isReady || isStreaming}
          style={styles.input}
        />
        <button
          type="submit"
          disabled={!isReady || isStreaming || !input.trim()}
          style={styles.sendBtn}
        >
          {isStreaming ? '...' : 'Send'}
        </button>
      </form>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    height: '500px',
    maxWidth: '600px',
    border: '1px solid #333',
    borderRadius: '8px',
    overflow: 'hidden',
    backgroundColor: '#1a1a2e',
    color: '#eee',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    padding: '12px 16px',
    backgroundColor: '#16213e',
    borderBottom: '1px solid #333',
  },
  title: {
    margin: 0,
    fontSize: '16px',
    fontWeight: 600,
    flex: 1,
  },
  status: {
    fontSize: '12px',
    color: '#888',
    marginRight: '12px',
  },
  clearBtn: {
    padding: '4px 12px',
    fontSize: '12px',
    border: '1px solid #444',
    borderRadius: '4px',
    backgroundColor: 'transparent',
    color: '#888',
    cursor: 'pointer',
  },
  messages: {
    flex: 1,
    overflowY: 'auto',
    padding: '16px',
  },
  emptyState: {
    textAlign: 'center',
    color: '#666',
    padding: '40px',
  },
  message: {
    marginBottom: '12px',
    padding: '12px',
    borderRadius: '8px',
    maxWidth: '85%',
  },
  userMessage: {
    backgroundColor: '#00d4ff',
    color: '#000',
    marginLeft: 'auto',
  },
  assistantMessage: {
    backgroundColor: '#2d2d44',
  },
  systemMessage: {
    backgroundColor: '#3d1a1a',
    color: '#ff6b6b',
    fontSize: '13px',
  },
  messageRole: {
    fontSize: '11px',
    fontWeight: 600,
    marginBottom: '4px',
    opacity: 0.7,
  },
  messageContent: {
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-word',
  },
  error: {
    padding: '8px 16px',
    backgroundColor: '#3d1a1a',
    color: '#ff6b6b',
    fontSize: '13px',
  },
  inputArea: {
    display: 'flex',
    padding: '12px',
    backgroundColor: '#0f0f23',
    borderTop: '1px solid #333',
  },
  input: {
    flex: 1,
    padding: '10px 14px',
    border: 'none',
    borderRadius: '4px 0 0 4px',
    backgroundColor: '#16213e',
    color: '#eee',
    fontSize: '14px',
    outline: 'none',
  },
  sendBtn: {
    padding: '10px 20px',
    border: 'none',
    borderRadius: '0 4px 4px 0',
    backgroundColor: '#00d4ff',
    color: '#000',
    fontWeight: 600,
    cursor: 'pointer',
  },
};

export default ChatComponent;
