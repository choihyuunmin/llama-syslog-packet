<template>
  <div class="app">
    <!-- 헤더 -->
    <header class="header">
      <h1>
        <FolderOpen class="w-5 h-5 text-blue-500" />
        SysPacket Analysis Tool
      </h1>
      <div class="header-actions">
        <button class="secondary">
          <Settings class="w-4 h-4" />
          Settings
        </button>
      </div>
    </header>

    <!-- 메인 컨텐츠 -->
    <main class="main">
      <div class="container">
        <!-- 왼쪽 패널: 파일 업로드 및 메타정보 -->
        <div class="panel">
          <h2>File Management</h2>
          
          <FileUpload @upload-success="fetchFiles" />
          <FileMeta :file="selectedFile" />
        </div>

        <!-- 중앙 패널: 채팅 인터페이스 -->
        <div class="panel">
          <h2>Chat</h2>
          
          <ChatMessages :messages="messages" />
          <ChatInput 
            :selected-file-id="selectedFile?.id"
            @message-sent="handleMessageSent"
            @visualization-update="updateVisualization"
          />
        </div>

        <!-- 오른쪽 패널: 시각화 결과 -->
        <div class="panel">
          <h2>Visualization</h2>
          
          <VisualizationPanel :data="visualizationData" />
        </div>
      </div>
    </main>

    <!-- 푸터 -->
    <footer class="footer">
      <Github class="w-4 h-4" />
      <p>© 2024 SysPacket. All rights reserved.</p>
    </footer>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { FolderOpen, Settings, Github } from 'lucide-vue-next'
import axios from 'axios'

// 컴포넌트 임포트
import FileUpload from './components/file/FileUpload.vue'
import FileMeta from './components/file/FileMeta.vue'
import ChatMessages from './components/chat/ChatMessages.vue'
import ChatInput from './components/chat/ChatInput.vue'
import VisualizationPanel from './components/visualization/VisualizationPanel.vue'

// 상태 관리
const files = ref([])
const selectedFile = ref(null)
const messages = ref([])
const visualizationData = ref(null)

// 파일 목록 가져오기
const fetchFiles = async () => {
  try {
    const response = await axios.get('/api/files')
    files.value = response.data
  } catch (error) {
    console.error('파일 목록 조회 실패:', error)
  }
}

// 파일 삭제
const deleteFile = async (fileId) => {
  try {
    await axios.delete(`/api/files/${fileId}`)
    await fetchFiles()
    if (selectedFile.value?.id === fileId) {
      selectedFile.value = null
    }
  } catch (error) {
    console.error('파일 삭제 실패:', error)
  }
}

// 파일 선택
const selectFile = async (fileId) => {
  try {
    const response = await axios.get(`/api/files/${fileId}/meta`)
    selectedFile.value = response.data
  } catch (error) {
    console.error('파일 메타정보 조회 실패:', error)
  }
}

// 메시지 처리
const handleMessageSent = (message) => {
  messages.value.push(message)
}

// 시각화 데이터 업데이트
const updateVisualization = (data) => {
  visualizationData.value = data
}

// 컴포넌트 마운트 시 초기화
onMounted(() => {
  fetchFiles()
})
</script>

<style scoped>
/* 기존 스타일 제거 - layout.css로 이동 */
</style> 