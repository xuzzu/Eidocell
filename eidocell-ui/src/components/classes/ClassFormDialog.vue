<script setup lang="ts">
import { ref } from 'vue'

const emit = defineEmits<{
  submit: [data: { name: string; color: string }]
}>()

const dialogRef = ref<HTMLDialogElement>()
const name = ref('')
const color = ref('#808080')

function randomColor(): string {
  const h = Math.floor(Math.random() * 360)
  const s = 55 + Math.floor(Math.random() * 30) // 55-85%
  const l = 45 + Math.floor(Math.random() * 15) // 45-60%
  // Convert HSL to hex
  const a = s / 100 * Math.min(l / 100, 1 - l / 100)
  const f = (n: number) => {
    const k = (n + h / 30) % 12
    const c = l / 100 - a * Math.max(Math.min(k - 3, 9 - k, 1), -1)
    return Math.round(255 * c).toString(16).padStart(2, '0')
  }
  return `#${f(0)}${f(8)}${f(4)}`
}

function open(existing?: { name: string; color: string }) {
  name.value = existing?.name ?? ''
  color.value = existing?.color ?? randomColor()
  dialogRef.value?.showModal()
}

function close() {
  dialogRef.value?.close()
}

function onSubmit() {
  if (!name.value.trim()) return
  emit('submit', { name: name.value.trim(), color: color.value })
  close()
}

defineExpose({ open, close })
</script>

<template>
  <dialog ref="dialogRef" class="modal">
    <div class="modal-box rounded-[2px] border border-neutral p-6 shadow-2xl bg-base-100 w-full max-w-sm">
      <h3 class="font-bold text-lg tracking-widest uppercase">{{ name ? 'Edit Class' : 'New Class' }}</h3>
      <form @submit.prevent="onSubmit" class="mt-6 space-y-4">
        <div class="form-control">
          <label class="label pb-1"><span class="label-text font-bold text-[10px] tracking-widest uppercase text-neutral/70">Name</span></label>
          <input v-model="name" type="text" placeholder="e.g. Elongated" class="input input-bordered rounded-[2px] w-full font-mono text-sm focus:outline-neutral" />
        </div>
        <div class="form-control">
          <label class="label pb-1"><span class="label-text font-bold text-[10px] tracking-widest uppercase text-neutral/70">Color</span></label>
          <div class="flex items-center gap-3">
            <input v-model="color" type="color" class="w-10 h-10 p-0.5 border border-base-300 rounded-[2px] cursor-pointer bg-transparent" />
            <span class="font-mono text-sm font-bold uppercase tracking-wider text-neutral/80">{{ color }}</span>
          </div>
        </div>
        <div class="modal-action mt-8">
          <button type="button" class="h-10 px-6 rounded-[2px] text-[11px] font-bold tracking-widest uppercase hover:bg-base-200 transition-colors" @click="close">Cancel</button>
          <button type="submit" class="h-10 px-8 rounded-[2px] bg-neutral text-neutral-content text-[11px] font-bold tracking-widest uppercase transition-opacity hover:opacity-80">Save</button>
        </div>
      </form>
    </div>
    <form method="dialog" class="modal-backdrop bg-neutral/20"><button class="cursor-default">close</button></form>
  </dialog>
</template>
