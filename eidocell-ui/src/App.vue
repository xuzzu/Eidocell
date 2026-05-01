<script setup lang="ts">
import BaseLayout from './components/BaseLayout.vue'
import NotificationStack from './components/common/NotificationStack.vue'
import { useSessionWatcher } from './composables/useSessionWatcher'
import { useNotificationsStore } from './stores/notifications'
import { onMounted } from 'vue'

useSessionWatcher()

const notifications = useNotificationsStore()
onMounted(() => {
  notifications.connect()
})
</script>

<template>
  <BaseLayout>
    <router-view v-slot="{ Component }">
      <transition name="fade" mode="out-in">
        <component :is="Component" />
      </transition>
    </router-view>
    <NotificationStack />
  </BaseLayout>
</template>

<style>
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.15s ease;
}
.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
