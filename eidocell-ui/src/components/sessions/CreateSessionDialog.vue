<script setup lang="ts">
import { ref } from 'vue'

const emit = defineEmits<{
  create: [data: { name: string }]
}>()

const dialogRef = ref<HTMLDialogElement>()
const name = ref('')
const error = ref('')

function open() {
  name.value = ''
  error.value = ''
  dialogRef.value?.showModal()
}

function close() {
  dialogRef.value?.close()
}

function onSubmit() {
  if (!name.value.trim()) {
    error.value = 'Session name is required'
    return
  }
  emit('create', { name: name.value.trim() })
  close()
}

defineExpose({ open, close })
</script>

<template>
  <dialog ref="dialogRef" class="modal">
    <div class="modal-box rounded-[2px] border border-neutral p-6 shadow-2xl bg-base-100">
      <h3 class="font-bold text-lg tracking-widest uppercase">New Session</h3>
      <p class="text-[11px] font-mono text-neutral/60 mt-2 tracking-wider">
        STEP 1 OF 2 — NAME THE SESSION; YOU'LL IMPORT DATA NEXT
      </p>
      <form @submit.prevent="onSubmit" class="mt-6 space-y-4">
        <div class="form-control">
          <label class="label pb-1"><span class="label-text font-bold text-[10px] tracking-widest uppercase text-neutral/70">Session Name</span></label>
          <input
            v-model="name"
            type="text"
            placeholder="e.g. Blood Sample Analysis"
            class="input input-bordered rounded-[2px] w-full font-mono text-sm focus:outline-neutral"
            autofocus
          />
        </div>
        <p v-if="error" class="text-error text-xs font-mono tracking-tight pt-2">{{ error }}</p>
        <div class="modal-action mt-8">
          <button type="button" class="h-10 px-6 rounded-[2px] text-[11px] font-bold tracking-widest uppercase hover:bg-base-200 transition-colors" @click="close">Cancel</button>
          <button type="submit" class="h-10 px-8 rounded-[2px] bg-neutral text-neutral-content text-[11px] font-bold tracking-widest uppercase transition-opacity hover:opacity-80">Continue to Import</button>
        </div>
      </form>
    </div>
    <form method="dialog" class="modal-backdrop bg-neutral/20">
      <button class="cursor-default">close</button>
    </form>
  </dialog>
</template>
