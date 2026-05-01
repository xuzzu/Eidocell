import { defineStore } from 'pinia'
import { ref } from 'vue'
import type { NotificationMessage } from '@/types'
import { BASE_URL } from '@/api/client'

export const useNotificationsStore = defineStore('notifications', () => {
  const notifications = ref<NotificationMessage[]>([])
  let eventSource: EventSource | null = null

  function connect() {
    if (eventSource) return
    eventSource = new EventSource(`${BASE_URL}/notifications/stream`)
    
    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        if (data.type === 'connected') return
        
        const notif: NotificationMessage = {
          id: data.timestamp + Math.random().toString(36).substr(2, 9),
          ...data
        }
        
        notifications.value.push(notif)
        
        // Auto remove success/info after 5s
        if (['success', 'info', 'warning', 'error'].includes(data.level)) {
            // Keep errors for 10s, others 4s
            setTimeout(() => {
                removeNotification(notif.id)
            }, data.level === 'error' ? 10000 : 4000)
        }

      } catch (err) {
        console.error('Error parsing notification', err)
      }
    }
    
    eventSource.onerror = () => {
      eventSource?.close()
      eventSource = null
      setTimeout(connect, 5000) // reconnect backoff
    }
  }

  function removeNotification(id: string) {
    const idx = notifications.value.findIndex(n => n.id === id)
    if (idx !== -1) {
      notifications.value.splice(idx, 1)
    }
  }

  return {
    notifications,
    connect,
    removeNotification
  }
})
