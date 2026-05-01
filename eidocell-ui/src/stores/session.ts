import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import * as api from '@/api/sessions'
import type { SessionListItem, SessionOut, SessionCreate } from '@/types'

const STORAGE_KEY = 'eidocell_current_session_id'

export const useSessionStore = defineStore('session', () => {
  const sessions = ref<SessionListItem[]>([])
  const currentSessionId = ref<string | null>(null)
  const currentSession = ref<SessionOut | null>(null)
  const loading = ref(false)

  const hasCurrentSession = computed(() => !!currentSessionId.value)

  async function fetchSessions() {
    loading.value = true
    try {
      sessions.value = await api.listSessions()
    } finally {
      loading.value = false
    }
  }

  async function createSession(data: SessionCreate) {
    const session = await api.createSession(data)
    await fetchSessions()
    return session
  }

  async function selectSession(id: string) {
    currentSessionId.value = id
    localStorage.setItem(STORAGE_KEY, id)
    currentSession.value = await api.getSession(id)
    await api.updateSession(id, {})
  }

  async function deleteSession(id: string) {
    await api.deleteSession(id)
    if (currentSessionId.value === id) {
      currentSessionId.value = null
      currentSession.value = null
      localStorage.removeItem(STORAGE_KEY)
    }
    await fetchSessions()
  }

  async function loadPersistedSession() {
    const storedId = localStorage.getItem(STORAGE_KEY)
    if (storedId) {
      try {
        currentSession.value = await api.getSession(storedId)
        currentSessionId.value = storedId
      } catch {
        currentSessionId.value = null
        currentSession.value = null
        localStorage.removeItem(STORAGE_KEY)
      }
    }
  }

  return {
    sessions, currentSessionId, currentSession, loading,
    hasCurrentSession,
    fetchSessions, createSession, selectSession, deleteSession, loadPersistedSession,
  }
})
