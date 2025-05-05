<template>
  <div class="chat-messages">
    <div class="messages-container" ref="messagesContainer">
      <div 
        v-for="(message, index) in messages" 
        :key="index"
        :class="['message', message.role]"
      >
        <div class="message-content">
          <div class="message-header">
            <span class="role">
              <component :is="message.role === 'user' ? User : Bot" class="w-4 h-4" />
              {{ message.role === 'user' ? '사용자' : 'AI' }}
            </span>
            <span class="time">{{ message.timestamp }}</span>
          </div>
          <div class="message-body">{{ message.content }}</div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, watch, nextTick } from 'vue'
import { User, Bot } from 'lucide-vue-next'

const props = defineProps({
  messages: {
    type: Array,
    required: true
  }
})

const messagesContainer = ref(null)

// 메시지가 변경될 때마다 스크롤을 아래로 내림
watch(() => props.messages.length, async () => {
  await nextTick() // DOM 업데이트를 기다림
  if (messagesContainer.value) {
    messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
  }
}, { immediate: true })
</script>

<style scoped>
.chat-messages {
  flex: 1;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.messages-container {
  flex: 1;
  overflow-y: auto;
  padding: 1.5rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
  scroll-behavior: smooth; /* 부드러운 스크롤 효과 */
}

.message {
  max-width: 80%;
  animation: fadeIn 0.3s ease-in-out;
}

.message.user {
  align-self: flex-end;
}

.message.assistant {
  align-self: flex-start;
}

.message-content {
  padding: 1rem;
  border-radius: 0.5rem;
  background-color: white;
  border: 1px solid #e2e8f0;
}

.message.user .message-content {
  background-color: #e0f2fe;
  border-color: #bae6fd;
}

.message-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
  font-size: 0.75rem;
  color: #64748b;
}

.role {
  display: flex;
  align-items: center;
  gap: 0.25rem;
}

.message-body {
  font-size: 0.875rem;
  line-height: 1.5;
  white-space: pre-wrap;
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
</style> 