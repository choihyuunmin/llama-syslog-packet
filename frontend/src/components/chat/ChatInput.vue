<template>
  <div class="chat-input">
    <div class="input-container">
      <textarea
        v-model="message"
        @keydown.enter.prevent="sendMessage"
        placeholder="메시지를 입력하세요..."
        rows="3"
      ></textarea>
      <button 
        @click="sendMessage"
        class="send-button"
      >
        <Send class="w-4 h-4" />
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { Send } from 'lucide-vue-next'
import axios from 'axios'

const emit = defineEmits(['message-sent'])

const message = ref('')
const selectedFileId = ref(null)

const sendMessage = async () => {
  if (!message.value.trim()) return

  // 사용자 메시지 먼저 표시
  emit('message-sent', {
    role: 'user',
    content: message.value,
    timestamp: new Date().toLocaleTimeString()
  })

  try {
    const response = await axios.post('/api/chat', {
      message: message.value,
      model: ""
    })

    // AI 응답 표시
    emit('message-sent', {
      role: 'assistant',
      content: response.data.response,
      timestamp: new Date().toLocaleTimeString()
    })

    if (response.data.visualization) {
      emit('visualization-update', response.data.visualization)
    }
  } catch (error) {
    console.error('메시지 전송 실패:', error)
    // 에러 메시지 표시
    emit('message-sent', {
      role: 'assistant',
      content: '죄송합니다. 응답을 생성하는 중에 오류가 발생했습니다.',
      timestamp: new Date().toLocaleTimeString()
    })
  }

  message.value = ''
}

defineProps({
  selectedFileId: {
    type: [String, Number],
    default: null
  }
})
</script>

<style scoped>
.chat-input {
  padding: 1rem;
  border-top: 1px solid #e2e8f0;
  background-color: white;
}

.input-container {
  display: flex;
  gap: 0.5rem;
  align-items: flex-end;
}

textarea {
  flex: 1;
  padding: 0.75rem;
  border: 1px solid #e2e8f0;
  border-radius: 0.375rem;
  font-size: 0.875rem;
  resize: none;
  background-color: white;
  min-height: 3rem;
  max-height: 8rem;
}

textarea:focus {
  outline: none;
  border-color: #3b82f6;
  box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.1);
}

.send-button {
  height: 2.5rem;
  width: 2.5rem;
  padding: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: #3b82f6;
  color: white;
  border: none;
  border-radius: 0.375rem;
  cursor: pointer;
  transition: all 0.2s;
}

.send-button:hover {
  background-color: #2563eb;
}
</style> 