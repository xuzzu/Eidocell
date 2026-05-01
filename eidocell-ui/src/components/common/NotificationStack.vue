<script setup lang="ts">
import { computed } from 'vue'
import { X, CheckCircle, AlertTriangle, Info, AlertCircle } from 'lucide-vue-next'
import { useNotificationsStore } from '@/stores/notifications'
import { useTasksStore } from '@/stores/tasks'
import { apiPost } from '@/api/client'

const notifStore = useNotificationsStore()
const taskStore = useTasksStore()

const taskList = computed(() => taskStore.activeTaskList)

async function cancelTask(taskId: string) {
  try {
    await apiPost(`/tasks/${taskId}/cancel`)
  } catch (e) {
    console.error("Failed to cancel task", e)
  }
}
</script>

<template>
  <div class="fixed bottom-4 right-4 z-[9999] flex flex-col gap-3 pointer-events-none items-end max-w-sm w-full">
    <!-- Tasks -->
    <transition-group name="list">
        <div
            v-for="task in taskList"
            :key="task.id"
            class="pointer-events-auto bg-base-100 border border-base-300 shadow-lg rounded-[2px] p-4 w-full flex flex-col gap-2"
        >
          <div class="flex items-center justify-between">
             <span class="text-[11px] font-bold tracking-widest uppercase text-neutral">
               {{ task.name }}
             </span>
             <button
               v-if="task.status === 'pending' || task.status === 'running'"
               class="text-neutral/40 hover:text-error transition-colors"
               title="Cancel Task"
               @click="cancelTask(task.id)"
             >
               <X class="w-3.5 h-3.5" />
             </button>
             <button
               v-else
               class="text-neutral/40 hover:text-neutral transition-colors"
               title="Dismiss"
               @click="taskStore.clearCompleted()"
             >
               <X class="w-3.5 h-3.5" />
             </button>
          </div>

          <div class="flex items-center gap-2 text-[10px] font-mono text-neutral/70">
            <span v-if="task.status === 'running' || task.status === 'pending'" class="loading loading-spinner loading-xs"></span>
            <CheckCircle v-else-if="task.status === 'completed'" class="w-3.5 h-3.5 text-success" />
            <AlertTriangle v-else-if="task.status === 'failed'" class="w-3.5 h-3.5 text-error" />
            <Info v-else class="w-3.5 h-3.5 text-info" />
            <span class="flex-1 truncate">{{ task.message || task.status }}</span>
            <span v-if="task.status === 'running'">{{ Math.round(task.percentage) }}%</span>
          </div>

          <div v-if="task.status === 'running'" class="h-1 bg-base-200 overflow-hidden w-full">
             <div class="h-full bg-neutral transition-all duration-300" :style="{ width: `${task.percentage}%` }"></div>
          </div>
        </div>

        <!-- Generic Notifications -->
        <div
            v-for="notif in notifStore.notifications"
            :key="notif.id"
            class="pointer-events-auto bg-base-100 border border-base-300 shadow-lg rounded-[2px] p-3 w-full flex items-start gap-3"
            :class="{
               'border-l-4 border-l-success': notif.level === 'success',
               'border-l-4 border-l-error': notif.level === 'error',
               'border-l-4 border-l-warning': notif.level === 'warning',
               'border-l-4 border-l-info': notif.level === 'info',
            }"
        >
            <CheckCircle v-if="notif.level === 'success'" class="w-4 h-4 text-success shrink-0 mt-0.5" />
            <AlertCircle v-else-if="notif.level === 'error'" class="w-4 h-4 text-error shrink-0 mt-0.5" />
            <AlertTriangle v-else-if="notif.level === 'warning'" class="w-4 h-4 text-warning shrink-0 mt-0.5" />
            <Info v-else class="w-4 h-4 text-info shrink-0 mt-0.5" />

            <div class="flex-1 flex flex-col min-w-0">
               <span class="text-[11px] font-bold tracking-widest uppercase text-neutral truncate">
                  {{ notif.title }}
               </span>
               <span v-if="notif.message" class="text-[10px] font-mono text-neutral/70 mt-1 break-words">
                  {{ notif.message }}
               </span>
            </div>
            
            <button
               class="text-neutral/40 hover:text-neutral transition-colors shrink-0"
               @click="notifStore.removeNotification(notif.id)"
            >
               <X class="w-3.5 h-3.5" />
            </button>
        </div>
    </transition-group>
  </div>
</template>

<style scoped>
.list-enter-active,
.list-leave-active {
  transition: all 0.3s ease;
}
.list-enter-from {
  opacity: 0;
  transform: translateX(30px);
}
.list-leave-to {
  opacity: 0;
  transform: translateX(30px);
}
</style>
