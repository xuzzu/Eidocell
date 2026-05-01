<script setup lang="ts">
import { ref } from 'vue'

defineProps<{
  title: string
  message: string
  confirmLabel?: string
  danger?: boolean
}>()

const emit = defineEmits<{
  confirm: []
  cancel: []
}>()

const dialogRef = ref<HTMLDialogElement>()

function open() {
  dialogRef.value?.showModal()
}

function close() {
  dialogRef.value?.close()
}

function onConfirm() {
  close()
  emit('confirm')
}

function onCancel() {
  close()
  emit('cancel')
}

defineExpose({ open, close })
</script>

<template>
  <dialog ref="dialogRef" class="modal">
    <div class="modal-box">
      <h3 class="font-bold text-lg">{{ title }}</h3>
      <p class="py-4">{{ message }}</p>
      <div class="modal-action">
        <button class="btn btn-ghost" @click="onCancel">Cancel</button>
        <button
          class="btn"
          :class="danger ? 'btn-error' : 'btn-primary'"
          @click="onConfirm"
        >
          {{ confirmLabel ?? 'Confirm' }}
        </button>
      </div>
    </div>
    <form method="dialog" class="modal-backdrop">
      <button @click="onCancel">close</button>
    </form>
  </dialog>
</template>
