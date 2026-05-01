export interface NotificationMessage {
  id: string
  type: string
  level: 'info' | 'success' | 'warning' | 'error'
  title: string
  message: string
  data: any
  timestamp: number
}
