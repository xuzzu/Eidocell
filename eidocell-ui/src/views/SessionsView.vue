<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { Plus, Folder, X } from 'lucide-vue-next'
import { useSessionStore } from '@/stores/session'
import { getPreviewStatus } from '@/api/sessions'
import type { PreviewStatus } from '@/types'
import SessionCard from '@/components/sessions/SessionCard.vue'
import CreateSessionDialog from '@/components/sessions/CreateSessionDialog.vue'
import ConfirmDialog from '@/components/common/ConfirmDialog.vue'
import EmptyState from '@/components/common/EmptyState.vue'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'

const router = useRouter()
const sessionStore = useSessionStore()
const createDialog = ref<InstanceType<typeof CreateSessionDialog>>()
const confirmDialog = ref<InstanceType<typeof ConfirmDialog>>()
const deleteTargetId = ref<string | null>(null)
const errorMessage = ref('')

const openingSessionId = ref<string | null>(null)
const openingStatus = ref<PreviewStatus | null>(null)
let pollHandle: number | null = null

onMounted(() => {
  sessionStore.fetchSessions()
})

onUnmounted(() => {
  stopPolling()
})

async function onCreate(data: { name: string }) {
  errorMessage.value = ''
  try {
    const session = await sessionStore.createSession(data)
    router.push({ name: 'import-wizard', params: { sessionId: session.id } })
  } catch (e: any) {
    errorMessage.value = e.message || 'Failed to create session'
  }
}

function stopPolling() {
  if (pollHandle !== null) {
    clearInterval(pollHandle)
    pollHandle = null
  }
}

async function actuallyOpen(id: string) {
  await sessionStore.selectSession(id)
  router.push('/workspace/gallery')
}

async function onSelect(id: string) {
  errorMessage.value = ''
  try {
    const status = await getPreviewStatus(id)
    if (status.ready) {
      await actuallyOpen(id)
      return
    }
    // Block on previews. Show modal + poll.
    openingSessionId.value = id
    openingStatus.value = status
    pollHandle = window.setInterval(async () => {
      if (!openingSessionId.value) return
      try {
        const next = await getPreviewStatus(openingSessionId.value)
        openingStatus.value = next
        if (next.ready) {
          stopPolling()
          const sid = openingSessionId.value
          openingSessionId.value = null
          openingStatus.value = null
          if (sid) await actuallyOpen(sid)
        } else if (next.phase === 'failed') {
          stopPolling()
          errorMessage.value = next.message || 'Session is not ready'
          openingSessionId.value = null
          openingStatus.value = null
        }
      } catch {
        // transient
      }
    }, 500) as unknown as number
  } catch (e: any) {
    errorMessage.value = e.message || 'Failed to open session'
  }
}

function cancelOpening() {
  stopPolling()
  openingSessionId.value = null
  openingStatus.value = null
}

function onDeleteRequest(id: string) {
  deleteTargetId.value = id
  confirmDialog.value?.open()
}

async function onDeleteConfirm() {
  if (deleteTargetId.value) {
    errorMessage.value = ''
    try {
      await sessionStore.deleteSession(deleteTargetId.value)
    } catch (e: any) {
      errorMessage.value = e.message || 'Failed to delete session'
    }
    deleteTargetId.value = null
  }
}
</script>

<template>
  <div class="p-8 max-w-5xl mx-auto flex flex-col h-full">
    <div class="flex items-center justify-between mb-8 pb-4 border-b border-base-300 shrink-0">
      <div>
        <h1 class="text-2xl font-bold tracking-widest uppercase">Sessions</h1>
        <p class="text-[11px] font-mono text-neutral/50 mt-1 tracking-wider">CREATE OR OPEN AN IMAGE ANALYSIS SESSION</p>
      </div>
      <button class="h-9 px-4 flex items-center gap-2 rounded-[2px] bg-neutral text-neutral-content text-[11px] font-bold tracking-widest uppercase transition-opacity hover:opacity-80 shadow-sm" @click="createDialog?.open()">
        <Plus class="w-4 h-4" />
        New Session
      </button>
    </div>

    <div v-if="sessionStore.loading" class="flex justify-center py-16">
      <LoadingSpinner size="loading-lg" />
    </div>

    <EmptyState
      v-else-if="sessionStore.sessions.length === 0"
      :icon="Folder"
      title="No sessions yet"
      description="Create a new session to start analyzing your images"
      action-label="New Session"
      @action="createDialog?.open()"
    />

    <div v-else class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 auto-rows-max overflow-y-auto pr-2 pb-8">
      <SessionCard
        v-for="session in sessionStore.sessions"
        :key="session.id"
        :session="session"
        @select="onSelect"
        @delete="onDeleteRequest"
      />
    </div>

    <div v-if="errorMessage" class="alert alert-error rounded-[2px] mb-4 text-sm font-mono">{{ errorMessage }}</div>

    <CreateSessionDialog ref="createDialog" @create="onCreate" />
    <ConfirmDialog
      ref="confirmDialog"
      title="Delete Session"
      message="Are you sure? This will permanently delete the session and all its data."
      confirm-label="Delete"
      :danger="true"
      @confirm="onDeleteConfirm"
    />

    <!-- Preview-pregen blocking modal -->
    <div
      v-if="openingSessionId && openingStatus && !openingStatus.ready"
      class="fixed inset-0 z-[700] bg-base-100/80 backdrop-blur-sm flex items-center justify-center"
    >
      <div class="w-full max-w-md bg-base-100 border border-base-300 rounded-[2px] shadow-xl p-6 space-y-4">
        <div class="flex items-center justify-between">
          <h2 class="text-[12px] font-bold tracking-widest uppercase">Preparing session</h2>
          <button
            class="w-7 h-7 flex items-center justify-center rounded-[2px] hover:bg-base-200 transition-colors"
            @click="cancelOpening"
            title="Cancel"
          ><X class="w-4 h-4" /></button>
        </div>
        <p class="text-[11px] font-mono text-neutral/70 tracking-tight">
          {{ openingStatus.phase === 'importing'
            ? 'Import still running.'
            : openingStatus.phase === 'pregenerating'
              ? 'Pre-generating per-channel previews.'
              : openingStatus.phase === 'failed'
                ? 'Session preparation failed.'
                : 'Working...' }}
        </p>
        <progress
          class="progress progress-neutral w-full rounded-[2px]"
          :value="openingStatus.progress" max="100"
        />
        <p class="text-[10px] font-mono text-neutral/60 truncate">{{ openingStatus.message }}</p>
      </div>
    </div>
  </div>
</template>
