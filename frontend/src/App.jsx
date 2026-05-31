import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "./index.css";

const APP_MODE = window.APP_MODE || "chat";
const IS_DEV_MODE = APP_MODE === "dev";
const AUTH_EXPIRED_EVENT = "openlawai:auth-expired";
const MAX_SSE_RECONNECT_ATTEMPTS = 5;
const MAX_CONVERSATION_DETAIL_MESSAGES = 500;
const UPLOAD_RETRY_ATTEMPTS = 3;
const JSON_REQUEST_TIMEOUT_MS = 30000;
const UPLOAD_TIMEOUT_MS = 120000;
const SSE_CONNECT_TIMEOUT_MS = 30000;
const SSE_IDLE_TIMEOUT_MS = 120000;

const getSafeMarkdownHref = (href) => {
  if (typeof href !== "string") {
    return null;
  }
  const trimmed = href.trim();
  if (!trimmed) {
    return null;
  }
  if (trimmed.startsWith("/") && !trimmed.startsWith("//")) {
    return trimmed;
  }
  try {
    const parsed = new URL(trimmed, window.location.origin);
    if (parsed.protocol === "http:" || parsed.protocol === "https:") {
      return trimmed;
    }
  } catch {
    return null;
  }
  return null;
};

const notifyApiBoundary = (status, responseData = {}) => {
  if (status === 401) {
    window.dispatchEvent(new CustomEvent(AUTH_EXPIRED_EVENT));
    return;
  }
};

const createAbortContext = (parentSignal, timeoutMs, timeoutMessage) => {
  const controller = new AbortController();
  const onParentAbort = () => controller.abort(parentSignal.reason);
  if (parentSignal) {
    if (parentSignal.aborted) {
      controller.abort(parentSignal.reason);
    } else {
      parentSignal.addEventListener("abort", onParentAbort, { once: true });
    }
  }
  const timeoutError = new Error(timeoutMessage);
  timeoutError.name = "TimeoutError";
  const timeoutId = window.setTimeout(() => controller.abort(timeoutError), timeoutMs);
  return {
    controller,
    signal: controller.signal,
    clearConnectTimeout: () => window.clearTimeout(timeoutId),
    cleanup: () => {
      window.clearTimeout(timeoutId);
      if (parentSignal) {
        parentSignal.removeEventListener("abort", onParentAbort);
      }
    },
  };
};

const fetchJSON = async (url, options = {}) => {
  const {
    headers: optionHeaders = {},
    signal: parentSignal,
    timeoutMs = JSON_REQUEST_TIMEOUT_MS,
    ...fetchOptions
  } = options;
  const headers = {
    "Content-Type": "application/json",
    "X-CSRFToken": window.CSRF_TOKEN ?? "",
    ...optionHeaders,
  };
  const abortContext = createAbortContext(
    parentSignal,
    timeoutMs,
    "The request took too long.",
  );
  try {
    const response = await fetch(url, {
      credentials: "include",
      ...fetchOptions,
      headers,
      signal: abortContext.signal,
    });
    let data;
    if (response.ok) {
      try {
        data = await response.json();
      } catch {
        throw new Error("Ugyldig svar fra serveren.");
      }
    } else {
      data = await response.json().catch(() => ({}));
    }
    if (!response.ok) {
      notifyApiBoundary(response.status, data);
      const err = new Error(data.detail || data.error || "Something went wrong");
      err.responseData = data;
      err.status = response.status;
      throw err;
    }
    return data;
  } catch (err) {
    if (abortContext.signal.aborted && abortContext.signal.reason) {
      throw abortContext.signal.reason;
    }
    throw err;
  } finally {
    abortContext.cleanup();
  }
};

const fetchUploadJSON = async (url, formData, options = {}) => {
  const abortContext = createAbortContext(
    options.signal,
    options.timeoutMs ?? UPLOAD_TIMEOUT_MS,
    "Upload took too long.",
  );
  try {
    const response = await fetch(url, {
      credentials: "include",
      method: "POST",
      headers: {
        "X-CSRFToken": window.CSRF_TOKEN ?? "",
        ...(options.headers || {}),
      },
      body: formData,
      ...options,
      signal: abortContext.signal,
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      notifyApiBoundary(response.status, data);
      const err = new Error(data.detail || data.error || "Something went wrong");
      err.responseData = data;
      err.status = response.status;
      throw err;
    }
    return data;
  } catch (err) {
    if (abortContext.signal.aborted && abortContext.signal.reason) {
      throw abortContext.signal.reason;
    }
    throw err;
  } finally {
    abortContext.cleanup();
  }
};

/**
 * Fetch SSE stream from the chat API.
 * @param {string} url - The URL to fetch
 * @param {object} options - Fetch options (method, body, etc.)
 * @param {object} callbacks - Event callbacks
 * @param {function} callbacks.onStatus - Called with status updates
 * @param {function} callbacks.onChunk - Called with text chunks
 * @param {function} callbacks.onDone - Called with final data
 * @param {function} callbacks.onError - Called with error details
 * @param {function} callbacks.onNotice - Called with notice details
 */
const fetchSSE = async (url, options, callbacks) => {
  const headers = {
    "Content-Type": "application/json",
    "X-CSRFToken": window.CSRF_TOKEN ?? "",
    // Don't set Accept header - DRF's content negotiation doesn't handle text/event-stream
    ...(options.headers || {}),
  };
  const abortContext = createAbortContext(
    options.signal,
    options.connectTimeoutMs ?? SSE_CONNECT_TIMEOUT_MS,
    "Tilkoblingen til serveren tok for lang tid.",
  );
  let reader;
  try {
    const response = await fetch(url, {
      credentials: "include",
      ...options,
      headers,
      signal: abortContext.signal,
    });

    if (!response.ok) {
      const text = await response.text();
      let detail = "Something went wrong";
      let responseData = {};
      try {
        responseData = JSON.parse(text);
        detail = responseData.detail || responseData.error || detail;
      } catch {
        // ignore parse error
      }
      notifyApiBoundary(response.status, responseData);
      const err = new Error(detail);
      err.status = response.status;
      err.responseData = responseData;
      throw err;
    }

    abortContext.clearConnectTimeout();
    reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let currentEvent = null;
    let currentData = [];

    const dispatchEvent = (event, data) => {
      try {
        const parsed = JSON.parse(data);
        switch (event) {
          case "status":
            callbacks.onStatus?.(parsed);
            break;
          case "chunk":
            callbacks.onChunk?.(parsed);
            break;
          case "done":
            callbacks.onDone?.(parsed);
            break;
          case "error":
            callbacks.onError?.(parsed);
            break;
          case "notice":
            callbacks.onNotice?.(parsed);
            break;
        }
      } catch {
        callbacks.onError?.({ detail: "Could not read server response." });
        return false;
      }
      return true;
    };

    const flushEvent = () => {
      if (!currentEvent) {
        return true;
      }
      const ok = dispatchEvent(currentEvent, currentData.join("\n"));
      currentEvent = null;
      currentData = [];
      return ok;
    };

    while (true) {
      let idleTimeoutId;
      try {
        const readResult = await Promise.race([
          reader.read(),
          new Promise((_, reject) => {
            idleTimeoutId = window.setTimeout(() => {
              const timeoutError = new Error("Tilkoblingen til serveren ble inaktiv.");
              timeoutError.name = "TimeoutError";
              abortContext.controller.abort(timeoutError);
              reject(timeoutError);
            }, options.idleTimeoutMs ?? SSE_IDLE_TIMEOUT_MS);
          }),
        ]);
        window.clearTimeout(idleTimeoutId);
        const { done, value } = readResult;
        if (done) {
          if (buffer.trim()) {
            const lines = buffer.split("\n");
            for (const line of lines) {
              if (line.startsWith("event: ")) {
                currentEvent = line.slice(7).trim();
              } else if (line.startsWith("data: ")) {
                currentData.push(line.slice(6));
              } else if (line === "" && currentEvent) {
                if (!flushEvent()) {
                  return;
                }
              }
            }
          }
          if (currentEvent && !flushEvent()) {
            return;
          }
          callbacks.onStreamEnd?.();
          break;
        }

        const chunk = decoder.decode(value, { stream: true });
        buffer += chunk;
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("event: ")) {
            currentEvent = line.slice(7).trim();
          } else if (line.startsWith("data: ")) {
            currentData.push(line.slice(6));
          } else if (line === "" && currentEvent) {
            if (!flushEvent()) {
              return;
            }
          }
        }
      } catch (err) {
        if (idleTimeoutId !== undefined) {
          window.clearTimeout(idleTimeoutId);
        }
        throw err;
      }
    }
  } catch (err) {
    if (abortContext.signal.aborted && abortContext.signal.reason) {
      throw abortContext.signal.reason;
    }
    throw err;
  } finally {
    abortContext.cleanup();
    if (reader) {
      reader.releaseLock();
    }
  }
};

const App = () => {
  const [view, setView] = useState("loading");
  const [user, setUser] = useState(null);
  const [conversations, setConversations] = useState([]);
  const [activeLocalId, setActiveLocalId] = useState(null);
  const [statusMessage, setStatusMessage] = useState(null);
  const [statusTone, setStatusTone] = useState("info");
  const [input, setInput] = useState("");
  const [qualityMode, setQualityMode] = useState("thorough"); // "fast" or "thorough"
  const [hasBootstrappedChat, setHasBootstrappedChat] = useState(false);
  const [hasLoadedConversations, setHasLoadedConversations] = useState(false);
  const [hasMoreConversations, setHasMoreConversations] = useState(false);
  const [isLoadingMoreConversations, setIsLoadingMoreConversations] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(() => window.innerWidth > 768);
  const [activeDialog, setActiveDialog] = useState(null);
  const dialogResolveRef = useRef(null);
  const composerRef = useRef(null);
  const messagesEndRef = useRef(null);
  const isLoadingMoreRef = useRef(false);
  const requestControllersRef = useRef(new Map());
  const activeSendLocalIdsRef = useRef(new Set());
  const sseControllersRef = useRef(new Map());
  const sseReconnectTimersRef = useRef(new Map());
  const subscribeToEventsRef = useRef(null);
  const conversationsRef = useRef([]);

  const showDialog = useCallback((options) => {
    return new Promise((resolve) => {
      dialogResolveRef.current = resolve;
      setActiveDialog(options);
    });
  }, []);

  const resolveDialog = useCallback((value) => {
    dialogResolveRef.current?.(value);
    dialogResolveRef.current = null;
    setActiveDialog(null);
  }, []);


  const loadConversationSummaries = useCallback(
    async (append = false, currentConversations = []) => {
      if (append && isLoadingMoreRef.current) return;

      const offset = append ? currentConversations.length : 0;
      if (append) {
        isLoadingMoreRef.current = true;
        setIsLoadingMoreConversations(true);
      }

      try {
        const data = await fetchJSON(`/api/conversations/?limit=20&offset=${offset}`, { method: "GET" });
        const mapped = (data.conversations || []).map(mapServerConversationSummary);

        if (append) {
          setConversations((prev) => [...prev, ...mapped]);
        } else {
          mapped.sort((a, b) => b.updatedAt - a.updatedAt);
          setConversations(mapped);
          if (mapped.length > 0) {
            setActiveLocalId((current) => current ?? mapped[0].localId);
          }
        }
        setHasMoreConversations(data.has_more ?? false);
      } catch (err) {
        if (!append) setConversations([]);
        setStatusTone("error");
        setStatusMessage(err.message);
      } finally {
        setHasLoadedConversations(true);
        if (append) {
          isLoadingMoreRef.current = false;
          setIsLoadingMoreConversations(false);
        }
      }
    },
    [setStatusTone, setStatusMessage]
  );

  const loadMoreConversations = useCallback(() => {
    if (hasMoreConversations && !isLoadingMoreRef.current) {
      loadConversationSummaries(true, conversations);
    }
  }, [hasMoreConversations, loadConversationSummaries, conversations]);

  useEffect(() => {
    activeSendLocalIdsRef.current = new Set(
      conversations.filter((conv) => conv.isSending).map((conv) => conv.localId)
    );
  }, [conversations]);

  const viewRef = useRef(view);
  viewRef.current = view;
  conversationsRef.current = conversations;


  const resetWorkspace = useCallback(() => {
    requestControllersRef.current.forEach((controller) => controller.abort());
    requestControllersRef.current.clear();
    // Abort all active SSE subscriptions
    sseControllersRef.current.forEach((controller) => controller.abort());
    sseControllersRef.current.clear();
    sseReconnectTimersRef.current.forEach((timeoutId) => window.clearTimeout(timeoutId));
    sseReconnectTimersRef.current.clear();

    setConversations([]);
    setActiveLocalId(null);
    setStatusMessage(null);
    setStatusTone("info");
    setInput("");
    setHasBootstrappedChat(false);
    setHasLoadedConversations(false);
  }, []);

  useEffect(() => {
    const requestControllers = requestControllersRef.current;
    const controllers = sseControllersRef.current;
    const reconnectTimers = sseReconnectTimersRef.current;
    return () => {
      requestControllers.forEach((controller) => controller.abort());
      requestControllers.clear();
      controllers.forEach((controller) => controller.abort());
      controllers.clear();
      reconnectTimers.forEach((timeoutId) => window.clearTimeout(timeoutId));
      reconnectTimers.clear();
    };
  }, []);

  useEffect(() => {
    const handleAuthExpired = () => {
      resetWorkspace();
      setUser(null);
      setView("login");
      setStatusTone("error");
      setStatusMessage("Your session has expired. Please log in again.");
    };
    window.addEventListener(AUTH_EXPIRED_EVENT, handleAuthExpired);
    return () => {
      window.removeEventListener(AUTH_EXPIRED_EVENT, handleAuthExpired);
    };
  }, [resetWorkspace]);

  useEffect(() => {
    const bootstrap = async () => {
      try {
        const data = await fetchJSON("/api/auth/session/", { method: "GET" });
        window.CSRF_TOKEN = data.csrf_token;
        if (data.authenticated) {
          setUser(data.user);
          setView("chat");
        } else {
          resetWorkspace();
          setUser(null);
          setView("login");
        }
      } catch (err) {
        resetWorkspace();
        setUser(null);
        setView("login");
      }
    };
    bootstrap();
  }, [resetWorkspace]);

  useEffect(() => {
    if (view === "chat" && user && !hasLoadedConversations) {
      loadConversationSummaries();
    }
  }, [view, user, hasLoadedConversations, loadConversationSummaries]);


  useEffect(() => {
    if (view === "chat" && hasLoadedConversations && !hasBootstrappedChat && conversations.length === 0) {
      const blank = createBlankConversation();
      setConversations([blank]);
      setActiveLocalId(blank.localId);
      setHasBootstrappedChat(true);
    }
  }, [view, hasLoadedConversations, hasBootstrappedChat, conversations.length]);

  const activeConversation = conversations.find((c) => c.localId === activeLocalId) || null;
  const activeMessages = activeConversation?.messages || [];
  const activeIsSending = activeConversation?.isSending ?? false;
  const activeStreamingStatus = activeConversation?.streamingStatus ?? null;
  const activeStreamingText = activeConversation?.streamingText ?? "";

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [activeLocalId, activeMessages.length, activeIsSending]);

  const ensureConversationMessages = useCallback(
    async (conversation, _retryCount = 0) => {
      if (!conversation?.serverId || conversation.hasLoadedMessages || conversation.isLoadingMessages) {
        return;
      }
      setConversations((prev) =>
        prev.map((conv) =>
          conv.localId === conversation.localId ? { ...conv, isLoadingMessages: true } : conv
        )
      );
      try {
        const detailPath = IS_DEV_MODE
          ? `/api/dev/conversations/${conversation.serverId}/`
          : `/api/conversations/${conversation.serverId}/`;
        const data = await fetchJSON(
          `${detailPath}?limit=${MAX_CONVERSATION_DETAIL_MESSAGES}&offset=0`,
          { method: "GET" }
        );
        const payloadMessages = data.messages || data.ui_messages || [];
        const mappedMessages = mapServerMessages(payloadMessages);
        const metadataPayload = data.metadata || data.conversation?.metadata || conversation.metadata || {};
        const llmHistory = data.llm_history || metadataPayload.llm_history || [];
        const lastMsg = mappedMessages[mappedMessages.length - 1];
        const isAwaitingAssistantReply = Boolean(lastMsg && lastMsg.role === "user");
        const shouldAttemptReconnect =
          isAwaitingAssistantReply && !conversation.isSending && !conversation.hasAttemptedReconnect;
        setConversations((prev) =>
          prev.map((conv) =>
            conv.localId === conversation.localId
              ? {
                  ...conv,
                  messages: mappedMessages,
                  hasLoadedMessages: true,
                  isLoadingMessages: false,
                  title: data.conversation?.title || deriveTitle(mappedMessages),
                  lastMessage: data.conversation?.last_message || deriveLastMessage(mappedMessages),
                  updatedAt: parseTimestamp(data.conversation?.updated_at),
                  metadata: metadataPayload,
                  devData: buildDevData(metadataPayload, llmHistory),
                  hasAttemptedReconnect: shouldAttemptReconnect
                    ? true
                    : (isAwaitingAssistantReply ? conv.hasAttemptedReconnect : false),
                }
              : conv
          )
        );

        // Reconnect: if last message is from the user (no AI response yet),
        // processing may still be active — subscribe to events
        if (shouldAttemptReconnect) {
          void subscribeToEventsRef.current(conversation.serverId, conversation.localId);
        }
      } catch (err) {
        if (_retryCount < 3) {
          await new Promise((r) => setTimeout(r, 1000 * (_retryCount + 1)));
          return ensureConversationMessages(conversation, _retryCount + 1);
        }
        setConversations((prev) =>
          prev.map((conv) =>
            conv.localId === conversation.localId ? { ...conv, isLoadingMessages: false } : conv
          )
        );
        setStatusTone("error");
        setStatusMessage(err.message);
      }
    },
    [setStatusMessage, setStatusTone]
  );

  useEffect(() => {
    if (view === "chat" && activeLocalId) {
      const conversation = conversationsRef.current.find((conv) => conv.localId === activeLocalId);
      if (conversation) {
        ensureConversationMessages(conversation);
      }
    }
  }, [view, activeLocalId, ensureConversationMessages]);

  // Auto-reconnect SSE when returning from phone sleep / app switch
  useEffect(() => {
    const handleVisibility = () => {
      if (document.hidden) return;

      // Clear stale error messages from before the app was backgrounded
      setStatusMessage(null);
      setStatusTone("info");

      if (viewRef.current !== "chat" || !activeLocalId) return;
      const conversation = conversationsRef.current.find((c) => c.localId === activeLocalId);
      if (!conversation?.serverId) return;

      if (conversation.isSending) {
        // The OS likely killed the SSE connection while backgrounded.
        // Abort the stale SSE and any pending reconnect timer, then start
        // a fresh subscription from attempt 0 so the user isn't penalized.
        const reconnectTimer = sseReconnectTimersRef.current.get(conversation.serverId);
        if (reconnectTimer !== undefined) {
          window.clearTimeout(reconnectTimer);
          sseReconnectTimersRef.current.delete(conversation.serverId);
        }
        const existing = sseControllersRef.current.get(conversation.serverId);
        if (existing) existing.abort();
        void subscribeToEventsRef.current(conversation.serverId, conversation.localId);
        return;
      }

      if (!conversation.hasLoadedMessages) {
        ensureConversationMessages(conversation);
      }
    };
    document.addEventListener("visibilitychange", handleVisibility);
    return () => document.removeEventListener("visibilitychange", handleVisibility);
  }, [activeLocalId, ensureConversationMessages]);

  const handleAuthSuccess = (userData) => {
    resetWorkspace();
    setUser(userData);
    setView("chat");
  };

  const handleLogout = async () => {
    try {
      await fetchJSON("/api/auth/logout/", { method: "POST" });
      resetWorkspace();
      setUser(null);
      setView("login");
    } catch (err) {
      setStatusTone("error");
      setStatusMessage(err.message || "Could not log out.");
    }
  };


  const handleNewChat = () => {
    const existingBlank = conversations.find(
      (conv) =>
        !conv.serverId &&
        conv.messages.length === 0 &&
        (conv.pendingFiles || []).length === 0 &&
        (conv.documents || []).length === 0 &&
        !conv.isSending &&
        !conv.isUploading
    );
    if (existingBlank) {
      setActiveLocalId(existingBlank.localId);
      setInput("");
      setStatusMessage(null);
      setStatusTone("info");
      composerRef.current?.focus();
      return;
    }

    const fresh = createBlankConversation();
    setConversations((prev) => [fresh, ...prev]);
    setActiveLocalId(fresh.localId);
    setInput("");
    setStatusMessage(null);
    setStatusTone("info");
    composerRef.current?.focus();
  };

  const touchConversation = useCallback(async (serverId) => {
    if (!serverId) return;
    await fetchJSON(`/api/conversations/${serverId}/`, { method: "PATCH", body: "{}" });
  }, []);

  const handleSelectConversation = (localId) => {
    const target = conversations.find((c) => c.localId === localId);
    setActiveLocalId(localId);
    setStatusMessage(null);
    setInput("");
    composerRef.current?.focus();

    if (target?.serverId) {
      touchConversation(target.serverId);
      setConversations((prev) => {
        const current = prev.find((c) => c.localId === localId);
        if (!current?.serverId) {
          return prev;
        }
        return promoteConversation(prev, { ...current, updatedAt: Date.now() });
      });
    }
  };


  const handleRenameConversation = async (localId) => {
    const target = conversations.find((c) => c.localId === localId);
    if (!target?.serverId) return;
    const newTitle = await showDialog({
      type: "prompt",
      title: "Rename",
      message: "New name for the conversation:",
      defaultValue: target.title || "",
      confirmLabel: "Save",
    });
    if (!newTitle || !newTitle.trim()) return;
    try {
      const data = await fetchJSON(`/api/conversations/${target.serverId}/`, {
        method: "PATCH",
        body: JSON.stringify({ title: newTitle.trim() }),
      });
      const updatedTitle = data.conversation?.title || newTitle.trim();
      setConversations((prev) =>
        prev.map((conv) => (conv.localId === target.localId ? { ...conv, title: updatedTitle } : conv))
      );
      setStatusMessage(null);
    } catch (err) {
      setStatusTone("error");
      setStatusMessage(err.message);
    }
  };

  const handleDeleteConversation = async (localId) => {
    const target = conversations.find((c) => c.localId === localId);
    if (!target) return;

    if (!target.serverId) {
      setConversations((prev) => {
        const next = prev.filter((c) => c.localId !== target.localId);
        if (localId === activeLocalId) {
          if (next.length === 0) {
            const fresh = createBlankConversation();
            setActiveLocalId(fresh.localId);
            return [fresh];
          }
          setActiveLocalId(next[0].localId);
        }
        return next;
      });
      setStatusMessage(null);
      return;
    }

    const confirmDelete = await showDialog({
      type: "confirm",
      title: "Delete conversation",
      message: "Delete this conversation?",
      confirmLabel: "Delete",
      confirmDanger: true,
    });
    if (!confirmDelete) return;
    try {
      await fetchJSON(`/api/conversations/${target.serverId}/`, { method: "DELETE" });
      setConversations((prev) => {
        const next = prev.filter((c) => c.localId !== target.localId);
        if (localId === activeLocalId) {
          if (next.length === 0) {
            const fresh = createBlankConversation();
            setActiveLocalId(fresh.localId);
            return [fresh];
          }
          setActiveLocalId(next[0].localId);
        }
        return next;
      });
      setStatusMessage(null);
    } catch (err) {
      setStatusTone("error");
      setStatusMessage(err.message);
    }
  };

  const handleAttachFile = (file) => {
    if (!activeConversation) return;
    const targetLocalId = activeConversation.localId;
    const pendingFile = createPendingFile(file);

    // Add file to pendingFiles (not uploaded yet)
    setConversations((prev) =>
      prev.map((conv) =>
        conv.localId === targetLocalId
          ? {
              ...conv,
              pendingFiles: [...(conv.pendingFiles || []), pendingFile],
            }
          : conv
      )
    );
  };

  const handleRemovePendingFile = (pendingFileId) => {
    if (!activeConversation) return;
    const targetLocalId = activeConversation.localId;

    setConversations((prev) =>
      prev.map((conv) =>
        conv.localId === targetLocalId
          ? {
              ...conv,
              pendingFiles: (conv.pendingFiles || []).filter((entry) => entry.id !== pendingFileId),
            }
          : conv
      )
    );
  };

  const uploadPendingFiles = async (pendingFiles, serverId, signal) => {
    if (pendingFiles.length === 0) return;

    for (let index = 0; index < pendingFiles.length; index += 1) {
      const pendingFile = pendingFiles[index];
      let uploadError = null;

      for (let attempt = 0; attempt < UPLOAD_RETRY_ATTEMPTS; attempt += 1) {
        const formData = new FormData();
        formData.append("file", pendingFile.file);
        try {
          await fetchUploadJSON(`/api/conversations/${serverId}/documents/`, formData, { signal });
          uploadError = null;
          break;
        } catch (err) {
          uploadError = err;
          if (attempt + 1 < UPLOAD_RETRY_ATTEMPTS) {
            await new Promise((resolve) => window.setTimeout(resolve, 1000 * (attempt + 1)));
          }
        }
      }

      if (uploadError) {
        uploadError.failedFiles = pendingFiles.slice(index);
        throw uploadError;
      }
    }
  };

  const loadConversationDocuments = useCallback(async (conversation) => {
    if (!conversation?.serverId) return;
    try {
      const data = await fetchJSON(`/api/conversations/${conversation.serverId}/documents/`, {
        method: "GET",
      });
      setConversations((prev) =>
        prev.map((conv) =>
          conv.localId === conversation.localId
            ? { ...conv, documents: data.documents || [] }
            : conv
        )
      );
    } catch (err) {
      setStatusTone("info");
      setStatusMessage("Could not load the document list right now.");
    }
  }, [setStatusMessage, setStatusTone]);

  const subscribeToEvents = useCallback(
    async (serverId, targetLocalId, reconnectAttempt = 0) => {
      // Single-subscription guard: abort any existing SSE for this conversation
      const reconnectTimer = sseReconnectTimersRef.current.get(serverId);
      if (reconnectTimer !== undefined) {
        window.clearTimeout(reconnectTimer);
        sseReconnectTimersRef.current.delete(serverId);
      }
      const existing = sseControllersRef.current.get(serverId);
      if (existing) existing.abort();

      const controller = new AbortController();
      sseControllersRef.current.set(serverId, controller);
      const clearController = () => {
        if (sseControllersRef.current.get(serverId) === controller) {
          sseControllersRef.current.delete(serverId);
        }
      };

      // Set initial streaming state
      setConversations((prev) =>
        prev.map((conv) =>
          conv.localId === targetLocalId
            ? {
                ...conv,
                isSending: true,
                streamingStatus: reconnectAttempt === 0
                  ? "Processing request..."
                  : "Reconnecting...",
                streamingText: reconnectAttempt === 0 ? "" : conv.streamingText,
              }
            : conv
        )
      );

      try {
        await fetchSSE(
          `/api/conversations/${serverId}/events/`,
          { method: "GET", signal: controller.signal },
          {
            onChunk: ({ text, is_catchup }) => {
              setConversations((prev) =>
                prev.map((conv) =>
                  conv.localId === targetLocalId
                    ? {
                        ...conv,
                        streamingStatus: null,
                        streamingText: is_catchup ? text : conv.streamingText + text,
                      }
                    : conv
                )
              );
            },
            onStatus: ({ message }) => {
              setConversations((prev) =>
                prev.map((conv) =>
                  conv.localId === targetLocalId
                    ? { ...conv, streamingStatus: message }
                    : conv
                )
              );
            },
            onDone: (data) => {
              clearController();
              const reconnectTimerId = sseReconnectTimersRef.current.get(serverId);
              if (reconnectTimerId !== undefined) {
                window.clearTimeout(reconnectTimerId);
                sseReconnectTimersRef.current.delete(serverId);
              }
              const doneStatus = data.status;


              if (doneStatus === "stalled") {
                setConversations((prev) =>
                  prev.map((conv) =>
                    conv.localId === targetLocalId
                      ? { ...conv, isSending: false, streamingStatus: null, streamingText: "", hasLoadedMessages: false }
                      : conv
                  )
                );
                setStatusTone("info");
                setStatusMessage("Processing may still be ongoing. Refresh the page in a moment.");
                return;
              }

              // completed or not_processing — finalize
              setConversations((prev) => {
                const targetConv = prev.find((c) => c.localId === targetLocalId);
                if (!targetConv) return prev;

                let updatedMessages = targetConv.messages;
                if (targetConv.streamingText && doneStatus === "completed") {
                  updatedMessages = [
                    ...targetConv.messages,
                    {
                      id: `ai-${Date.now()}`,
                      role: "assistant",
                      content: targetConv.streamingText,
                      metadata: {},
                    },
                  ];
                }

                const updated = {
                  ...targetConv,
                  messages: updatedMessages,
                  isSending: false,
                  isLoadingMessages: false,
                  streamingStatus: null,
                  streamingText: "",
                  hasLoadedMessages: false,
                  documents: [],
                };
                return promoteConversation(prev, updated);
              });

              // The state update above sets hasLoadedMessages: false, but no
              // existing useEffect re-fires to trigger the DB reload (deps are
              // stable). Schedule an explicit reload so the definitive server
              // data (including any answer completed while backgrounded) appears.
              setTimeout(() => {
                const conv = conversationsRef.current.find((c) => c.localId === targetLocalId);
                if (conv && !conv.hasLoadedMessages && !conv.isLoadingMessages) {
                  void ensureConversationMessages(conv);
                }
              }, 50);
            },
            onNotice: ({ message: noticeMessage }) => {
              setStatusTone("info");
              setStatusMessage(noticeMessage);
            },
            onError: ({ detail }) => {
              clearController();
              const reconnectTimerId = sseReconnectTimersRef.current.get(serverId);
              if (reconnectTimerId !== undefined) {
                window.clearTimeout(reconnectTimerId);
                sseReconnectTimersRef.current.delete(serverId);
              }
              setConversations((prev) =>
                prev.map((conv) =>
                  conv.localId === targetLocalId
                    ? { ...conv, isSending: false, streamingStatus: null, streamingText: "" }
                    : conv
                )
              );
              setStatusTone("error");
              setStatusMessage(detail || "Something went wrong.");
            },
            onStreamEnd: () => {
              setConversations((prev) =>
                prev.map((conv) =>
                  conv.localId === targetLocalId && conv.isSending
                    ? { ...conv, isSending: false, streamingStatus: null, hasLoadedMessages: false }
                    : conv
                )
              );
            },
          }
        );
      } catch (err) {
        if (err.name === "AbortError") {
          clearController();
          return; // Intentional abort — ignore
        }
        clearController();
        if (reconnectAttempt >= MAX_SSE_RECONNECT_ATTEMPTS) {
          setConversations((prev) =>
            prev.map((conv) =>
              conv.localId === targetLocalId
                ? {
                    ...conv,
                    isSending: false,
                    streamingStatus: null,
                    hasLoadedMessages: false,
                    hasAttemptedReconnect: true,
                  }
                : conv
            )
          );
          setStatusTone("error");
          setStatusMessage("Connection lost. Please reload the conversation.");
          return;
        }

        const nextAttempt = reconnectAttempt + 1;
        const delayMs = Math.min(1000 * (2 ** reconnectAttempt), 15000);
        setConversations((prev) =>
          prev.map((conv) =>
            conv.localId === targetLocalId
              ? {
                  ...conv,
                  isSending: true,
                  streamingStatus: "Kobler til igjen...",
                  hasAttemptedReconnect: true,
                }
              : conv
          )
        );
        const timerId = window.setTimeout(() => {
          sseReconnectTimersRef.current.delete(serverId);
          void subscribeToEventsRef.current(serverId, targetLocalId, nextAttempt);
        }, delayMs);
        sseReconnectTimersRef.current.set(serverId, timerId);
      }
    },
    []
  );

  subscribeToEventsRef.current = subscribeToEvents;

  // Load documents when conversation is selected
  useEffect(() => {
    if (activeConversation?.serverId && activeConversation.documents === undefined) {
      loadConversationDocuments(activeConversation);
    }
  }, [activeConversation?.serverId, activeConversation?.documents, loadConversationDocuments]);


  const cancelActiveUpload = useCallback((localId) => {
    const controller = requestControllersRef.current.get(localId);
    if (controller) {
      controller.abort();
    }
  }, []);

  const handleSendMessage = async () => {
    if (!activeConversation || !input.trim() || activeConversation.isSending) return;
    if (activeSendLocalIdsRef.current.has(activeConversation.localId)) return;

    const trimmed = input.trim();
    const pendingId = `local-${Date.now()}`;
    const targetLocalId = activeConversation.localId;
    let targetServerId = activeConversation.serverId;
    const pendingFilesToUpload = activeConversation.pendingFiles || [];
    const hasPendingFiles = pendingFilesToUpload.length > 0;

    // Snapshot pending files for display on the message bubble
    const attachedDocs = pendingFilesToUpload.map(entry => ({ filename: entry.file.name }));

    const optimisticMessage = {
      id: pendingId,
      role: "user",
      content: trimmed,
      metadata: attachedDocs.length > 0 ? { attached_documents: attachedDocs } : {},
    };
    const requestController = new AbortController();

    activeSendLocalIdsRef.current.add(targetLocalId);
    requestControllersRef.current.set(targetLocalId, requestController);

    // Set isSending and add optimistic message in one update
    setConversations((prev) =>
      prev.map((conv) =>
        conv.localId === targetLocalId
          ? {
              ...conv,
              messages: [...conv.messages, optimisticMessage],
              lastMessage: trimmed,
              isSending: true,
              streamingStatus: hasPendingFiles ? "Uploading documents..." : null,
              streamingText: "",
              hasAttemptedReconnect: false,
              documents: [],
              pendingFiles: [],
              isUploading: hasPendingFiles,
            }
          : conv
      )
    );

    setInput("");
    setStatusMessage(null);

    try {
      // If there are pending files, ensure conversation exists and upload them first
      if (hasPendingFiles) {
        if (!targetServerId) {
          const createData = await fetchJSON("/api/conversations/", { method: "POST" });
          targetServerId = createData.conversation?.id;
          if (!targetServerId) {
            throw new Error("Could not create conversation");
          }
          setConversations((prev) =>
            prev.map((conv) =>
              conv.localId === targetLocalId
                ? { ...conv, serverId: targetServerId }
                : conv
            )
          );
        }
        await uploadPendingFiles(pendingFilesToUpload, targetServerId, requestController.signal);
        setConversations((prev) =>
          prev.map((conv) =>
            conv.localId === targetLocalId
              ? { ...conv, isUploading: false }
              : conv
          )
        );
      }

      // Step 1: POST to start processing — returns 202 JSON
      const payload = {
        message: trimmed,
        conversation_id: targetServerId,
        quality_mode: qualityMode,
      };

      const result = await fetchJSON("/api/chat/", {
        method: "POST",
        body: JSON.stringify(payload),
        signal: requestController.signal,
      });

      const serverId = result.conversation_id;

      // Immediately update serverId from POST response
      setConversations((prev) =>
        prev.map((conv) =>
          conv.localId === targetLocalId ? { ...conv, serverId } : conv
        )
      );

      // Step 2: Subscribe to SSE events (fire-and-forget — manages its own lifecycle)
      void subscribeToEvents(serverId, targetLocalId);

    } catch (err) {
      // POST failed — reconcile
      const errorServerId = err.responseData?.conversation_id;
      const failedFiles = err.failedFiles || pendingFilesToUpload;

      if (errorServerId) {
        // Server created the conversation but processing failed — reload from DB
        setConversations((prev) =>
          prev.map((conv) =>
            conv.localId === targetLocalId
              ? {
                  ...conv,
                  serverId: errorServerId,
                  hasLoadedMessages: false,
                  isSending: false,
                  streamingStatus: null,
                  streamingText: "",
                  isUploading: false,
                  pendingFiles: failedFiles,
                }
              : conv
          )
        );
      } else if (targetServerId) {
        // Had a serverId already — reload from DB
        setConversations((prev) =>
          prev.map((conv) =>
            conv.localId === targetLocalId
              ? {
                  ...conv,
                  hasLoadedMessages: false,
                  isSending: false,
                  streamingStatus: null,
                  streamingText: "",
                  isUploading: false,
                  pendingFiles: failedFiles,
                }
              : conv
          )
        );
      } else {
        // No server identity — remove optimistic message
        setConversations((prev) =>
          prev.map((conv) =>
            conv.localId === targetLocalId
              ? {
                  ...conv,
                  messages: conv.messages.filter((msg) => msg.id !== pendingId),
                  isSending: false,
                  streamingStatus: null,
                  streamingText: "",
                  isUploading: false,
                  pendingFiles: failedFiles,
                }
              : conv
          )
        );
      }

      if (errorServerId || targetServerId) {
        await loadConversationDocuments({
          localId: targetLocalId,
          serverId: errorServerId || targetServerId,
        });
      }

      if (err.name === "AbortError") {
        setStatusTone("info");
        setStatusMessage("Upload was interrupted.");
      } else {
        setStatusTone("error");
        setStatusMessage(err.message);
      }
    } finally {
      requestControllersRef.current.delete(targetLocalId);
    }
  };

  const content = useMemo(() => {
    switch (view) {
      case "loading":
        return <div className="shell">Laster inn...</div>;
      case "login":
        return <LoginPanel onSuccess={handleAuthSuccess} onSwitch={() => setView("register")} />;
      case "register":
        return <RegisterPanel onSuccess={handleAuthSuccess} onSwitch={() => setView("login")} />;
      case "overview":
        return <OverviewPanel onBack={() => setView("chat")} />;
      case "chat":
        if (!activeConversation) {
          return <div className="shell">Starting conversation...</div>;
        }
        return (
          <div className="app-frame">
            <div className="chat-shell">
              <ChatSidebar
                conversations={conversations}
                activeLocalId={activeLocalId}
                onSelect={handleSelectConversation}
                onNewChat={handleNewChat}
                onRename={handleRenameConversation}
                onDelete={handleDeleteConversation}
                onLoadMore={loadMoreConversations}
                hasMore={hasMoreConversations}
                isLoadingMore={isLoadingMoreConversations}
                isOpen={sidebarOpen}
                onClose={() => setSidebarOpen(false)}
              />

              <main className="chat-main">
                <div className="chat-surface">
                  <ChatHeader
                    user={user}
                    onLogout={handleLogout}
                    onOverview={() => setView("overview")}
                    onToggleSidebar={() => setSidebarOpen((prev) => !prev)}
                    sidebarOpen={sidebarOpen}
                  />
                  <StatusBadge message={statusMessage} tone={statusTone} />

                  <section className="messages-panel">
                    {activeMessages.length === 0 && !activeIsSending ? (
                      <EmptyState />
                    ) : (
                      <div className="message-stream">
                        {activeMessages.map((message) => (
                          <MessageBubble key={message.id} message={message} />
                        ))}
                        {(activeIsSending || activeStreamingText) && (
                          <TypingIndicator
                            statusText={activeStreamingStatus}
                            streamingText={activeStreamingText}
                          />
                        )}
                        <div ref={messagesEndRef} />
                      </div>
                    )}
                  </section>

                  <ChatComposer
                    value={input}
                    onChange={setInput}
                    onSend={handleSendMessage}
                    disableInput={!activeConversation}
                    disableSend={!activeConversation || activeIsSending || !input.trim() || input.length > MAX_MESSAGE_CHARS}
                    inputRef={composerRef}
                    qualityMode={qualityMode}
                    onQualityModeChange={setQualityMode}
                    pendingFiles={activeConversation?.pendingFiles || []}
                    onAttachFile={handleAttachFile}
                    onRemovePendingFile={handleRemovePendingFile}
                    onCancelUpload={() => activeConversation && cancelActiveUpload(activeConversation.localId)}
                    isUploading={activeConversation?.isUploading || false}
                    conversationId={activeConversation?.serverId}
                    localId={activeLocalId}
                    onError={(msg) => {
                      setStatusTone("error");
                      setStatusMessage(msg);
                    }}
                  />
                </div>
              </main>
            </div>
          </div>
        );
      default:
        return null;
    }
  }, [
    view,
    conversations,
    activeConversation,
    activeLocalId,
    activeMessages,
    activeIsSending,
    activeStreamingStatus,
    activeStreamingText,
    input,
    qualityMode,
    statusMessage,
    statusTone,
    sidebarOpen,
    user,
  ]);

  return (
    <>
      {content}
      {activeDialog && (
        <AppDialog dialog={activeDialog} onResolve={resolveDialog} />
      )}
    </>
  );
};

const ChatSidebar = ({
  conversations,
  activeLocalId,
  onSelect,
  onNewChat,
  onRename,
  onDelete,
  onLoadMore,
  hasMore,
  isLoadingMore,
  isOpen,
  onClose,
}) => {
  const [openMenuId, setOpenMenuId] = useState(null);
  const listRef = useRef(null);

  const checkNeedMore = useCallback(() => {
    const el = listRef.current;
    if (!el || !hasMore || isLoadingMore) return;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 100;
    if (nearBottom) onLoadMore();
  }, [hasMore, isLoadingMore, onLoadMore]);

  useEffect(() => {
    checkNeedMore();
  }, [checkNeedMore, conversations.length]);

  const toggleMenu = (id) => {
    setOpenMenuId((current) => (current === id ? null : id));
  };

  const handleRename = (id) => {
    setOpenMenuId(null);
    onRename(id);
  };

  const handleDelete = (id) => {
    setOpenMenuId(null);
    onDelete(id);
  };

  const handleSelect = (localId) => {
    onSelect(localId);
    if (window.innerWidth <= 768) onClose();
  };

  const handleNewChat = () => {
    onNewChat();
    if (window.innerWidth <= 768) onClose();
  };

  return (
    <>
      {isOpen && <div className="sidebar-overlay" onClick={onClose} />}
      <aside className={`sidebar ${isOpen ? "open" : "closed"}`}>
        <div className="sidebar-header">
          <button className="sidebar-button wide" onClick={handleNewChat}>
            New conversation
          </button>
          <button className="sidebar-close-btn" onClick={onClose} aria-label="Close">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="15 18 9 12 15 6" />
            </svg>
          </button>
        </div>
        <div className="sidebar-list" ref={listRef} onScroll={checkNeedMore}>
          {conversations.map((conv) => (
            <div key={conv.localId} className={`chat-preview-row ${conv.localId === activeLocalId ? "active" : ""}`}>
              <button className="chat-preview" onClick={() => handleSelect(conv.localId)}>
                <div className="preview-title">{conv.title}</div>
              </button>
              <div className="preview-actions">
                <button className="ghost-icon" onClick={() => toggleMenu(conv.localId)} aria-label="More options">
                  ⋯
                </button>
                {openMenuId === conv.localId && (
                  <div className="preview-menu">
                    <button onClick={() => handleRename(conv.localId)}>Rename</button>
                    <button onClick={() => handleDelete(conv.localId)}>Delete</button>
                  </div>
                )}
              </div>
            </div>
          ))}
          {isLoadingMore && <div className="sidebar-loading">Laster flere...</div>}
        </div>
      </aside>
    </>
  );
};



const USAGE_PAGE_SIZE = 20;

const OverviewPanel = ({ onBack }) => {
  const [usageHistory, setUsageHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [usagePage, setUsagePage] = useState(0);
  const [hasMore, setHasMore] = useState(false);
  const [expandedId, setExpandedId] = useState(null);

  useEffect(() => {
    const loadUsage = async () => {
      setIsLoading(true);
      try {
        const offset = usagePage * USAGE_PAGE_SIZE;
        const data = await fetchJSON(`/api/usage/?limit=${USAGE_PAGE_SIZE}&offset=${offset}`, { method: "GET" });
        setUsageHistory(data.usage || []);
        setHasMore(data.has_more ?? false);
      } catch (err) {
        setUsageHistory([]);
        setHasMore(false);
      } finally {
        setIsLoading(false);
      }
    };
    loadUsage();
  }, [usagePage]);

  const formatDateTime = (isoString) => {
    const date = new Date(isoString);
    return date.toLocaleString("nb-NO", {
      day: "2-digit",
      month: "2-digit",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  return (
    <div className="overview-shell">
      <div className="overview-header">
        <button className="outline-button" onClick={onBack}>&larr; Back to chat</button>
        <h2>Overview</h2>
      </div>

      <div className="usage-history-section">
        <h3>Usage History</h3>
        {isLoading && <p className="muted">Loading...</p>}
        {!isLoading && usageHistory.length === 0 && (
          <p className="muted">No usage history yet.</p>
        )}
        {!isLoading && usageHistory.length > 0 && (
          <div className="usage-table-wrapper">
            <table className="usage-table">
              <thead>
                <tr>
                  <th></th>
                  <th>Date/time</th>
                  <th className="hide-mobile">Request</th>
                  <th>Tokens</th>
                </tr>
              </thead>
              <tbody>
                {usageHistory.map((item) => (
                  <Fragment key={item.id}>
                    <tr
                      className={`usage-row ${item.calls && item.calls.length > 0 ? "expandable" : ""} ${expandedId === item.id ? "expanded" : ""}`}
                      onClick={() => item.calls && item.calls.length > 0 && setExpandedId(expandedId === item.id ? null : item.id)}
                    >
                      <td className="expand-cell">
                        {item.calls && item.calls.length > 0 && (
                          <span className="expand-chevron">{expandedId === item.id ? "▾" : "▸"}</span>
                        )}
                      </td>
                      <td>{formatDateTime(item.created_at)}</td>
                      <td className="request-label hide-mobile">{item.request_label}</td>
                      <td>{(item.input_tokens + item.output_tokens).toLocaleString()}</td>
                    </tr>
                    {expandedId === item.id && (
                      <tr className="usage-label-mobile-row">
                        <td colSpan={4}>{item.request_label}</td>
                      </tr>
                    )}
                    {expandedId === item.id && item.calls && item.calls.map((call, idx) => (
                      <tr key={`${item.id}-call-${idx}`} className="usage-call-row">
                        <td></td>
                        <td className="call-model">{call.model}</td>
                        <td className="muted hide-mobile">In: {call.input_tokens.toLocaleString()} / Out: {call.output_tokens.toLocaleString()}</td>
                        <td>{(call.input_tokens + call.output_tokens).toLocaleString()}</td>
                      </tr>
                    ))}
                    {expandedId === item.id && item.services && item.services.map((svc) => (
                      <tr key={`${item.id}-svc-${svc}`} className="usage-call-row">
                        <td></td>
                        <td className="call-model">{svc}</td>
                        <td className="hide-mobile"></td>
                        <td className="service-tag">included</td>
                      </tr>
                    ))}
                  </Fragment>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {(usagePage > 0 || hasMore) && (
          <div className="usage-pagination">
            <button
              className="outline-button"
              onClick={() => setUsagePage((p) => p - 1)}
              disabled={usagePage === 0 || isLoading}
            >
              Previous
            </button>
            <span className="muted">Page {usagePage + 1}</span>
            <button
              className="outline-button"
              onClick={() => setUsagePage((p) => p + 1)}
              disabled={!hasMore || isLoading}
            >
              Next
            </button>
          </div>
        )}
      </div>
    </div>
  );
};


const ChatHeader = ({ user, onLogout, onOverview, onToggleSidebar, sidebarOpen }) => {
  const [menuOpen, setMenuOpen] = useState(false);
  const menuItemRefs = useRef([]);


  useEffect(() => {
    if (!menuOpen) {
      menuItemRefs.current = [];
      return;
    }
    const firstItem = menuItemRefs.current[0];
    firstItem?.focus();
  }, [menuOpen]);

  useEffect(() => {
    if (!menuOpen) return;
    const handleKeyDown = (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        setMenuOpen(false);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [menuOpen]);

  const registerMenuItem = (index) => (element) => {
    menuItemRefs.current[index] = element;
  };

  const handleMenuKeyDown = (event) => {
    const items = menuItemRefs.current.filter(Boolean);
    if (items.length === 0) return;
    const currentIndex = items.findIndex((item) => item === document.activeElement);
    if (event.key === "ArrowDown") {
      event.preventDefault();
      items[(currentIndex + 1 + items.length) % items.length].focus();
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      items[(currentIndex - 1 + items.length) % items.length].focus();
    } else if (event.key === "Home") {
      event.preventDefault();
      items[0].focus();
    } else if (event.key === "End") {
      event.preventDefault();
      items[items.length - 1].focus();
    } else if (event.key === "Escape") {
      event.preventDefault();
      setMenuOpen(false);
    }
  };

  return (
    <header className="chat-header">
      <div className="header-left">
        <button
          className={`sidebar-toggle ${sidebarOpen ? "hidden-toggle" : ""}`}
          onClick={onToggleSidebar}
          aria-label="Conversations"
          tabIndex={sidebarOpen ? -1 : 0}
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="9 18 15 12 9 6" />
          </svg>
        </button>
        <h1 id="text">OpenLawAI</h1>
      </div>
      <div className="header-menu-wrapper">
        <button
          className="burger-btn"
          onClick={() => setMenuOpen(!menuOpen)}
          aria-label="Meny"
          aria-expanded={menuOpen}
          aria-haspopup="menu"
          aria-controls="header-menu"
        >
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="3" y1="6" x2="21" y2="6" />
            <line x1="3" y1="12" x2="21" y2="12" />
            <line x1="3" y1="18" x2="21" y2="18" />
          </svg>
        </button>
        {menuOpen && (
          <>
            <div className="menu-backdrop" onClick={() => setMenuOpen(false)} />
            <div
              id="header-menu"
              className="header-dropdown"
              role="menu"
              onKeyDown={handleMenuKeyDown}
            >
              <button
                ref={registerMenuItem(0)}
                role="menuitem"
                onClick={() => { onOverview(); setMenuOpen(false); }}
              >
                Overview
              </button>
              <button
                ref={registerMenuItem(1)}
                role="menuitem"
                className="dropdown-danger"
                onClick={() => { onLogout(); setMenuOpen(false); }}
              >
                Log out
              </button>
            </div>
          </>
        )}
      </div>
    </header>
  );
};

const StatusBadge = ({ message, tone }) => {
  if (!message) return null;
  const className = tone === "error" ? "status-pill error" : "status-pill";
  return <div className={className}>{message}</div>;
};

const MessageBubble = ({ message }) => {
  const isUser = message.role === "user";
  const isInternal = message.metadata?.internal === true;
  const attachedDocs = message.metadata?.attached_documents || [];

  // In dev mode, show internal messages with special styling
  // In normal mode, internal messages are filtered out by the backend
  const rowClasses = [
    "message-row",
    isUser ? "message-user" : "message-ai",
    isInternal ? "message-internal" : "",
  ].filter(Boolean).join(" ");

  return (
    <div className={rowClasses}>
      {!isUser && (
        <div className={`message-avatar ${isInternal ? "internal" : ""}`}>
          {isInternal ? "🔧" : "AI"}
        </div>
      )}
      <div className={`message-card ${isInternal ? "internal" : ""}`}>
        {isInternal && IS_DEV_MODE && (
          <div className="internal-label">Internal message (not shown to user)</div>
        )}
        {attachedDocs.length > 0 && (
          <div className="message-attached-docs">
            {attachedDocs.map((doc, i) => (
              <span key={doc.id || i} className="attached-doc-chip">
                📄 {doc.filename}
              </span>
            ))}
          </div>
        )}
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          className="message-content"
          components={{
            a: ({ href, children }) => {
              // Check if this is a generated document download link
              const isDocumentDownload = href && href.startsWith("/api/documents/");
              if (isDocumentDownload) {
                return (
                  <a
                    href={href}
                    download
                    className="document-download-link"
                  >
                    <span className="download-icon">📄</span>
                    {children}
                    <span className="download-action">⬇️</span>
                  </a>
                );
              }
              const safeHref = getSafeMarkdownHref(href);
              if (!safeHref) {
                return <span>{children}</span>;
              }
              return (
                <a href={safeHref} target="_blank" rel="noopener noreferrer">
                  {children}
                </a>
              );
            },
          }}
        >
          {message.content}
        </ReactMarkdown>
      </div>
      {isUser && <div className="message-avatar user">DU</div>}
    </div>
  );
};

const TypingIndicator = ({ statusText, streamingText }) => {
  const [showAlternate, setShowAlternate] = useState(false);
  const [fadeClass, setFadeClass] = useState("fade-in");

  useEffect(() => {
    if (!statusText || streamingText) {
      setShowAlternate(false);
      setFadeClass("fade-in");
      return;
    }

    // Alternate between status text and "Venter..." every 3 seconds
    let fadeTimeoutId;
    const intervalId = setInterval(() => {
      setFadeClass("fade-out");
      fadeTimeoutId = window.setTimeout(() => {
        setShowAlternate((prev) => !prev);
        setFadeClass("fade-in");
      }, 300); // Match CSS transition duration
    }, 3000);

    return () => {
      clearInterval(intervalId);
      if (fadeTimeoutId !== undefined) {
        window.clearTimeout(fadeTimeoutId);
      }
    };
  }, [statusText, streamingText]);

  const displayText = showAlternate ? "Venter..." : statusText;

  return (
    <div className="typing-indicator-container" aria-live="polite">
      {statusText && !streamingText && (
        <div className={`status-text ${fadeClass}`}>{displayText}</div>
      )}
      {streamingText && (
        <div className="message-row message-ai">
          <div className="message-avatar">AI</div>
          <div className="message-card streaming">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              className="message-content"
              components={{
                a: ({ href, children }) => {
                  const safeHref = getSafeMarkdownHref(href);
                  if (!safeHref) {
                    return <span>{children}</span>;
                  }
                  return (
                    <a href={safeHref} target="_blank" rel="noopener noreferrer">
                      {children}
                    </a>
                  );
                },
              }}
            >
              {streamingText}
            </ReactMarkdown>
            <div className="streaming-cursor" />
          </div>
        </div>
      )}
    </div>
  );
};

const MAX_MESSAGE_CHARS = 40000;
const MAX_PENDING_FILES = 5;
const MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024;
const MAX_IMAGE_FILE_SIZE_BYTES = 50 * 1024 * 1024;
const ALLOWED_UPLOAD_TYPES = new Set([
  "application/pdf",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  "text/plain",
  "text/markdown",
  "image/png",
  "image/jpeg",
  "image/webp",
  "image/tiff",
]);
const ALLOWED_UPLOAD_EXTENSIONS = new Set([
  ".pdf",
  ".docx",
  ".txt",
  ".md",
  ".png",
  ".jpg",
  ".jpeg",
  ".webp",
  ".tif",
  ".tiff",
]);

const fileExtension = (name) => {
  const dot = name.lastIndexOf(".");
  return dot >= 0 ? name.slice(dot).toLowerCase() : "";
};

const validatePendingFile = (file) => {
  const extension = fileExtension(file.name);
  const isAllowedType = ALLOWED_UPLOAD_TYPES.has(file.type);
  const isAllowedExtension = ALLOWED_UPLOAD_EXTENSIONS.has(extension);
  if (!isAllowedType && !isAllowedExtension) {
    return "Ugyldig filtype.";
  }
  const isImage = file.type.startsWith("image/") || [".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"].includes(extension);
  const maxBytes = isImage ? MAX_IMAGE_FILE_SIZE_BYTES : MAX_FILE_SIZE_BYTES;
  if (file.size > maxBytes) {
    return `Filen er for stor. Maks ${Math.floor(maxBytes / (1024 * 1024))} MB.`;
  }
  return null;
};

const ChatComposer = ({
  value,
  onChange,
  onSend,
  disableInput,
  disableSend,
  inputRef,
  qualityMode,
  onQualityModeChange,
  pendingFiles,
  onAttachFile,
  onRemovePendingFile,
  onCancelUpload,
  isUploading,
  conversationId,
  localId,
  onError,
}) => {
  const fileInputRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);
  const canAttach = !!(conversationId || localId);

  useEffect(() => {
    if (!value && inputRef.current) {
      inputRef.current.style.height = 'auto';
    }
  }, [value]);

  const currentPendingCount = (pendingFiles || []).length;
  const charCount = value.length;
  const isNearLimit = charCount > MAX_MESSAGE_CHARS * 0.9;
  const isOverLimit = charCount > MAX_MESSAGE_CHARS;

  const handleFiles = (files) => {
    if (!canAttach || !onAttachFile) return;
    const validFiles = [];
    for (const file of files) {
      const validationError = validatePendingFile(file);
      if (validationError) {
        onError?.(`${file.name}: ${validationError}`);
        continue;
      }
      validFiles.push(file);
    }

    // Check file count limit
    const availableSlots = MAX_PENDING_FILES - currentPendingCount;
    if (availableSlots <= 0) {
      onError?.(`Maximum ${MAX_PENDING_FILES} files at a time. Remove some first.`);
      return;
    }

    const filesToAdd = validFiles.slice(0, availableSlots);
    if (filesToAdd.length < validFiles.length) {
      onError?.(`Bare ${availableSlots} fil(er) lagt til. Maks ${MAX_PENDING_FILES} filer om gangen.`);
    }

    for (const file of filesToAdd) {
      onAttachFile(file);
    }
  };

  const handleChange = (newValue) => {
    // Enforce character limit
    if (newValue.length <= MAX_MESSAGE_CHARS) {
      onChange(newValue);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files.length > 0) {
      handleFiles(Array.from(e.dataTransfer.files));
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    // Only set isDragging to false if we're actually leaving the drop zone
    // (not just moving to a child element)
    if (!e.currentTarget.contains(e.relatedTarget)) {
      setIsDragging(false);
    }
  };

  const hasPendingFiles = pendingFiles && pendingFiles.length > 0;

  return (
    <div className="composer">
      {hasPendingFiles && (
        <div className="composer-documents">
          {pendingFiles.map((pendingFile) => (
            <div key={pendingFile.id} className="document-chip pending">
              <span className="document-icon">📎</span>
              <span className="document-name">{pendingFile.file.name}</span>
              <span className="document-tokens">(venter)</span>
              <button
                className="document-remove"
                onClick={() => onRemovePendingFile?.(pendingFile.id)}
                title="Remove file"
              >
                ×
              </button>
            </div>
          ))}
        </div>
      )}
      {isUploading && (
        <div className="upload-progress">
          <span className="upload-spinner">⏳</span> Uploading document...
          <button type="button" className="composer-upload-cancel" onClick={onCancelUpload}>
            Cancel
          </button>
        </div>
      )}
      <div
        className={`composer-input-area ${isDragging ? "dragging" : ""}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
      >
        <textarea
          ref={inputRef}
          placeholder={isDragging ? "Drop file here..." : "Ask me anything..."}
          aria-label="Type your message"
          value={value}
          onChange={(e) => {
            handleChange(e.target.value);
            e.target.style.height = 'auto';
            e.target.style.height = Math.min(e.target.scrollHeight, 200) + 'px';
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              if (e.ctrlKey) {
                e.preventDefault();
                const start = e.target.selectionStart;
                const end = e.target.selectionEnd;
                const newValue = value.substring(0, start) + "\n" + value.substring(end);
                handleChange(newValue);
                requestAnimationFrame(() => {
                  e.target.selectionStart = e.target.selectionEnd = start + 1;
                  e.target.style.height = 'auto';
                  e.target.style.height = Math.min(e.target.scrollHeight, 200) + 'px';
                });
              } else {
                e.preventDefault();
                if (!disableSend && !isOverLimit) {
                  onSend();
                }
              }
            }
          }}
          disabled={disableInput}
          spellCheck={false}
        />
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.docx,.txt,.md,.png,.jpg,.jpeg,.webp,.tif,.tiff"
          style={{ display: "none" }}
          onChange={(e) => {
            if (e.target.files.length > 0) {
              handleFiles(Array.from(e.target.files));
              e.target.value = "";
            }
          }}
        />
        {/* Character counter */}
        {charCount > 0 && (
          <div className={`char-counter ${isNearLimit ? "warning" : ""} ${isOverLimit ? "error" : ""}`}>
            {charCount.toLocaleString()} / {MAX_MESSAGE_CHARS.toLocaleString()}
          </div>
        )}
      </div>
      <div className="composer-actions">
        <div className="composer-left">
          <div className="quality-toggle">
            <button
              className={`quality-btn ${qualityMode === "thorough" ? "active" : ""}`}
              onClick={() => onQualityModeChange("thorough")}
              title="More thorough"
              aria-label="Select thorough mode"
              aria-pressed={qualityMode === "thorough"}
            >
              Grundig
            </button>
            <button
              className={`quality-btn ${qualityMode === "fast" ? "active" : ""}`}
              onClick={() => onQualityModeChange("fast")}
              title="Faster"
              aria-label="Select fast mode"
              aria-pressed={qualityMode === "fast"}
            >
              Rask
            </button>
          </div>
          <button
            className="attach-button"
            onClick={() => fileInputRef.current?.click()}
            disabled={!canAttach || isUploading}
            title="Upload document (PDF, DOCX, TXT)"
            aria-label="Attach document"
          >
            📎
          </button>
        </div>
        <button className="primary-button" onClick={onSend} disabled={disableSend} aria-label="Send message">
          Send
        </button>
      </div>
    </div>
  );
};

const EmptyState = () => (
  <div className="empty-state">
    <div className="empty-card">
      <h3>Hei! Hva kan jeg hjelpe deg med i dag?</h3>
    </div>
  </div>
);


const AppDialog = ({ dialog, onResolve }) => {
  const [inputValue, setInputValue] = useState(dialog.defaultValue || "");
  if (!dialog) return null;

  const handleConfirm = () => {
    onResolve(dialog.type === "prompt" ? inputValue : true);
  };

  const handleCancel = () => {
    onResolve(dialog.type === "prompt" ? null : false);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter") handleConfirm();
    if (e.key === "Escape" && dialog.type !== "alert") handleCancel();
  };

  return (
    <div className="app-dialog-overlay" onKeyDown={handleKeyDown}>
      <div className="app-dialog" role="dialog" aria-modal="true" aria-labelledby="app-dialog-title">
        <h3 id="app-dialog-title">{dialog.title || (dialog.type === "alert" ? "Notice" : "Confirm")}</h3>
        <p className="app-dialog-message">{dialog.message}</p>
        {dialog.type === "prompt" && (
          <input
            className="app-dialog-input"
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            autoFocus
          />
        )}
        <div className="app-dialog-actions">
          {dialog.type !== "alert" && (
            <button className="outline-button" onClick={handleCancel}>
              {dialog.cancelLabel || "Cancel"}
            </button>
          )}
          <button
            className={dialog.confirmDanger ? "primary-button danger" : "primary-button"}
            onClick={handleConfirm}
            autoFocus={dialog.type !== "prompt"}
          >
            {dialog.confirmLabel || "OK"}
          </button>
        </div>
      </div>
    </div>
  );
};

const AuthShell = ({ title, subtitle, children, onSwitch, switchLabel }) => (
  <div className="shell auth-shell">
    <div className="auth-card">
      <div>
        <h2>{title}</h2>
        <p className="muted">{subtitle}</p>
      </div>
      {children}
      {onSwitch && switchLabel && (
        <button className="ghost-button" onClick={onSwitch}>
          {switchLabel}
        </button>
      )}
    </div>
  </div>
);

const LoginPanel = ({ onSuccess, onSwitch }) => {
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  const handleLogin = async (event) => {
    event.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const data = await fetchJSON("/api/auth/login/", {
        method: "POST",
        body: JSON.stringify({ username, password }),
      });
      if (data.success && onSuccess) onSuccess(data.user);
    } catch (err) {
      setError(err.message || "Invalid credentials.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <AuthShell
      title="Welcome to OpenLawAI"
      subtitle="Log in to chat with our legal AI assistant"
      onSwitch={onSwitch}
      switchLabel="Don't have an account? Register"
    >
      {error && <p className="error-text">{error}</p>}
      <form onSubmit={handleLogin} className="auth-form">
        <input
          type="text"
          placeholder="Username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          autoComplete="username"
          required
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          autoComplete="current-password"
          required
        />
        <button type="submit" className="primary-button" disabled={loading}>
          {loading ? "Logging in..." : "Log in"}
        </button>
      </form>
      <div className="login-info">
        <p className="muted">
          OpenLawAI helps you find relevant laws and regulations, analyze legal
          documents, and generate drafts of contracts and other legal documents.
          All answers must be verified by the user.
        </p>
      </div>
    </AuthShell>
  );
};

const RegisterPanel = ({ onSuccess, onSwitch }) => {
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [passwordConfirm, setPasswordConfirm] = useState("");

  const handleRegister = async (event) => {
    event.preventDefault();
    setError(null);
    if (password !== passwordConfirm) {
      setError("Passwords do not match.");
      return;
    }
    setLoading(true);
    try {
      const data = await fetchJSON("/api/auth/register/", {
        method: "POST",
        body: JSON.stringify({ username, password, password_confirm: passwordConfirm }),
      });
      if (data.success && onSuccess) onSuccess(data.user);
    } catch (err) {
      setError(err.message || "Registration failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <AuthShell
      title="Create Account"
      subtitle="Register to use OpenLawAI"
      onSwitch={onSwitch}
      switchLabel="Already have an account? Log in"
    >
      {error && <p className="error-text">{error}</p>}
      <form onSubmit={handleRegister} className="auth-form">
        <input
          type="text"
          placeholder="Username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          autoComplete="username"
          required
          minLength={3}
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          autoComplete="new-password"
          required
          minLength={8}
        />
        <input
          type="password"
          placeholder="Confirm password"
          value={passwordConfirm}
          onChange={(e) => setPasswordConfirm(e.target.value)}
          autoComplete="new-password"
          required
        />
        <button type="submit" className="primary-button" disabled={loading}>
          {loading ? "Creating account..." : "Register"}
        </button>
      </form>
    </AuthShell>
  );
};


function createBlankConversation() {
  return {
    localId: generateLocalId(),
    serverId: null,
    title: "New conversation",
    lastMessage: "",
    messages: [],
    updatedAt: Date.now(),
    hasLoadedMessages: true,
    isLoadingMessages: false,
    metadata: {},
    devData: buildDevData({}),
    isSending: false,
    streamingStatus: null,
    streamingText: "",
    hasAttemptedReconnect: false,
    documents: [],
    pendingFiles: [], // Files attached but not yet uploaded
    isUploading: false,
  };
}

function createPendingFile(file) {
  return {
    id: generateLocalId(),
    file,
  };
}

function generateLocalId() {
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function promoteConversation(list, updated) {
  const others = list.filter((conv) => conv.localId !== updated.localId);
  return [updated, ...others];
}

function mapServerMessages(messages = []) {
  return messages.map((msg, index) => ({
    id: msg.id ? String(msg.id) : `srv-${index}`,
    role: msg.role || "assistant",
    content: msg.content || "",
    metadata: msg.metadata || {},
  }));
}

function mapServerConversationSummary(conversation) {
  const serverId = conversation.id;
  return {
    localId: serverId,
    serverId,
    title: conversation.title || "New conversation",
    lastMessage: conversation.last_message || "",
    messages: [],
    updatedAt: parseTimestamp(conversation.updated_at),
    hasLoadedMessages: false,
    isLoadingMessages: false,
    metadata: conversation.metadata || {},
    devData: buildDevData(conversation.metadata || {}),
    isSending: false,
    streamingStatus: null,
    streamingText: "",
    hasAttemptedReconnect: false,
    documents: undefined, // Will be loaded when conversation is selected
    isUploading: false,
  };
}

function parseTimestamp(value) {
  if (!value) {
    return Date.now();
  }
  const parsed = Date.parse(value);
  return Number.isNaN(parsed) ? Date.now() : parsed;
}

function deriveTitle(messages) {
  const firstUserMsg = messages.find((msg) => msg.role === "user");
  if (!firstUserMsg) return "New conversation";
  return truncate(firstUserMsg.content, 32);
}

function deriveLastMessage(messages) {
  if (!messages.length) return "No messages yet";
  return truncate(messages[messages.length - 1].content, 48);
}

function truncate(text = "", length) {
  return text.length > length ? `${text.slice(0, length)}…` : text || "(empty message)";
}

function buildDevData(metadata = {}, overrideHistory) {
  const llmHistory = overrideHistory ?? metadata.llm_history ?? [];
  return {
    llmHistory,
    metadata,
  };
}

export default App;
