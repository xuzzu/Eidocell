<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { Plus, Folder } from 'lucide-vue-next'
import { useSessionStore } from '@/stores/session'
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

onMounted(() => {
  sessionStore.fetchSessions()
})

async function onCreate(data: { name: string; images_directory: string }) {
  errorMessage.value = ''
  try {
    const session = await sessionStore.createSession(data)
    await sessionStore.selectSession(session.id)
    router.push('/workspace/gallery')
  } catch (e: any) {
    errorMessage.value = e.message || 'Failed to create session'
  }
}

async function onSelect(id: string) {
  errorMessage.value = ''
  try {
    await sessionStore.selectSession(id)
    router.push('/workspace/gallery')
  } catch (e: any) {
    errorMessage.value = e.message || 'Failed to open session'
  }
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
  </div>
</template>
