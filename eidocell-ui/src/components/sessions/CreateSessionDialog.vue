<script setup lang="ts">
import { ref } from 'vue'
import { FolderOpen } from 'lucide-vue-next'

const emit = defineEmits<{
  create: [data: { name: string; images_directory: string }]
}>()

const dialogRef = ref<HTMLDialogElement>()
const name = ref('')
const imagesDirectory = ref('')
const error = ref('')

function open() {
  name.value = ''
  imagesDirectory.value = ''
  error.value = ''
  dialogRef.value?.showModal()
}

function close() {
  dialogRef.value?.close()
}

async function pickDirectory() {
  try {
    const dir = await window.ipcRenderer.invoke('select-directory')
    if (dir) {
      imagesDirectory.value = dir
      if (!name.value) {
        // Auto-fill name from directory name
        name.value = dir.split(/[/\\]/).pop() || ''
      }
    }
  } catch {
    // Fallback: let user type the path
  }
}

function onSubmit() {
  if (!name.value.trim()) {
    error.value = 'Session name is required'
    return
  }
  if (!imagesDirectory.value.trim()) {
    error.value = 'Images directory is required'
    return
  }
  emit('create', {
    name: name.value.trim(),
    images_directory: imagesDirectory.value.trim(),
  })
  close()
}

defineExpose({ open, close })
</script>

<template>
  <dialog ref="dialogRef" class="modal">
    <div class="modal-box rounded-[2px] border border-neutral p-6 shadow-2xl bg-base-100">
      <h3 class="font-bold text-lg tracking-widest uppercase">New Session</h3>
      <form @submit.prevent="onSubmit" class="mt-6 space-y-4">
        <div class="form-control">
          <label class="label pb-1"><span class="label-text font-bold text-[10px] tracking-widest uppercase text-neutral/70">Session Name</span></label>
          <input
            v-model="name"
            type="text"
            placeholder="e.g. Blood Sample Analysis"
            class="input input-bordered rounded-[2px] w-full font-mono text-sm focus:outline-neutral"
          />
        </div>
        <div class="form-control">
          <label class="label pb-1"><span class="label-text font-bold text-[10px] tracking-widest uppercase text-neutral/70">Images Directory</span></label>
          <div class="flex gap-2">
            <input
              v-model="imagesDirectory"
              type="text"
              placeholder="/path/to/images"
              class="input input-bordered rounded-[2px] flex-1 font-mono text-sm focus:outline-neutral"
            />
            <button type="button" class="h-12 w-12 flex items-center justify-center border border-base-300 rounded-[2px] hover:bg-base-200 transition-colors text-neutral shrink-0" @click="pickDirectory">
              <FolderOpen class="w-5 h-5 stroke-[1.5px]" />
            </button>
          </div>
        </div>
        <p v-if="error" class="text-error text-xs font-mono tracking-tight pt-2">{{ error }}</p>
        <div class="modal-action mt-8">
          <button type="button" class="h-10 px-6 rounded-[2px] text-[11px] font-bold tracking-widest uppercase hover:bg-base-200 transition-colors" @click="close">Cancel</button>
          <button type="submit" class="h-10 px-8 rounded-[2px] bg-neutral text-neutral-content text-[11px] font-bold tracking-widest uppercase transition-opacity hover:opacity-80">Create</button>
        </div>
      </form>
    </div>
    <form method="dialog" class="modal-backdrop bg-neutral/20">
      <button class="cursor-default">close</button>
    </form>
  </dialog>
</template>
