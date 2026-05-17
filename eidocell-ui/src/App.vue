<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import BaseLayout from './components/BaseLayout.vue'
import PopoutLayout from './components/PopoutLayout.vue'
import NotificationStack from './components/common/NotificationStack.vue'
import { useSessionWatcher } from './composables/useSessionWatcher'
import { useNotificationsStore } from './stores/notifications'
import { useSessionStore } from './stores/session'
import { onMounted } from 'vue'

const route = useRoute()
const isPopout = computed(() => route.meta.layout === 'popout')

if (isPopout.value) {
  // Popouts share localStorage with the main window. Seed the session id
  // synchronously so views that gate on `sessionStore.currentSessionId` (like
  // GalleryView's preview poll) start fetching immediately, then refresh the
  // session details asynchronously. windowSync keeps us in sync afterwards.
  const session = useSessionStore()
  const storedId = localStorage.getItem('eidocell_current_session_id')
  if (storedId) session.currentSessionId = storedId
  session.loadPersistedSession()
} else {
  useSessionWatcher()
}

const notifications = useNotificationsStore()
onMounted(() => {
  notifications.connect()
})
</script>

<template>
  <component :is="isPopout ? PopoutLayout : BaseLayout">
    <router-view v-slot="{ Component }">
      <transition name="fade" mode="out-in">
        <component :is="Component" />
      </transition>
    </router-view>
    <NotificationStack />
  </component>
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
