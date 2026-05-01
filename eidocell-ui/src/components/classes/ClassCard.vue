<script setup lang="ts">
import { computed } from 'vue'
import { Trash2, Users } from 'lucide-vue-next'
import { useSessionStore } from '@/stores/session'
import { classPreviewUrl } from '@/api/classes'
import type { ClassOut } from '@/types'

const props = defineProps<{ cls: ClassOut }>()
const emit = defineEmits<{
  select: [id: string]
  delete: [id: string]
}>()

const sessionStore = useSessionStore()
const previewSrc = computed(() =>
  classPreviewUrl(sessionStore.currentSessionId!, props.cls.id)
)
</script>

<template>
  <div
    class="card bg-base-100 shadow-sm border border-base-200 hover:shadow-md transition-shadow cursor-pointer"
    @click="emit('select', cls.id)"
  >
    <figure class="px-4 pt-4">
      <img :src="previewSrc" class="rounded-lg w-full h-32 object-cover bg-base-200" />
    </figure>
    <div class="card-body p-4 pt-3">
      <div class="flex items-center gap-2">
        <span class="w-3 h-3 rounded-full shrink-0" :style="{ backgroundColor: cls.color }"></span>
        <h3 class="font-semibold text-sm truncate">{{ cls.name }}</h3>
        <span class="ml-auto flex items-center gap-1 text-xs text-base-content/50">
          <Users class="w-3 h-3" />
          {{ cls.sample_count }}
        </span>
      </div>
      <div class="card-actions justify-end mt-2">
        <button
          v-if="cls.name !== 'Uncategorized'"
          class="btn btn-ghost btn-xs text-base-content/40 hover:text-error"
          @click.stop="emit('delete', cls.id)"
        >
          <Trash2 class="w-3.5 h-3.5" />
        </button>
      </div>
    </div>
  </div>
</template>
