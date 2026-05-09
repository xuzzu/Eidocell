<script setup lang="ts">
import { ref, nextTick } from 'vue'

const dialogRef = ref<HTMLDialogElement>()
const inputRef = ref<HTMLInputElement>()
const name = ref('')
const original = ref('')
let resolveFn: ((value: string | null) => void) | null = null

function open(initial: string): Promise<string | null> {
  name.value = initial
  original.value = initial
  dialogRef.value?.showModal()
  nextTick(() => {
    inputRef.value?.focus()
    inputRef.value?.select()
  })
  return new Promise(resolve => {
    resolveFn = resolve
  })
}

function submit() {
  const trimmed = name.value.trim()
  if (!trimmed || trimmed === original.value) {
    cancel()
    return
  }
  dialogRef.value?.close()
  resolveFn?.(trimmed)
  resolveFn = null
}

function cancel() {
  dialogRef.value?.close()
  resolveFn?.(null)
  resolveFn = null
}

defineExpose({ open })
</script>

<template>
  <dialog ref="dialogRef" class="modal" @close="cancel">
    <div class="modal-box max-w-sm rounded-[2px] p-5">
      <h3 class="text-[10px] font-bold tracking-widest uppercase text-neutral/60 mb-3">Rename Gate</h3>
      <form @submit.prevent="submit">
        <input
          ref="inputRef"
          v-model="name"
          type="text"
          class="input input-bordered rounded-[2px] w-full font-mono text-sm focus:outline-neutral"
          maxlength="80"
        />
        <div class="modal-action mt-4">
          <button type="button" class="btn btn-ghost btn-sm rounded-[2px]" @click="cancel">Cancel</button>
          <button
            type="submit"
            class="h-9 px-5 rounded-[2px] bg-neutral text-neutral-content text-[10px] font-bold tracking-widest uppercase hover:opacity-80 transition-opacity"
          >
            Rename
          </button>
        </div>
      </form>
    </div>
    <form method="dialog" class="modal-backdrop">
      <button @click="cancel">close</button>
    </form>
  </dialog>
</template>
