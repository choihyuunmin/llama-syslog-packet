<template>
  <div class="file-upload">
    <h3>File Upload</h3>
    <div class="upload-content">
      <div class="file-input">
        <button class="secondary w-full">
          <Upload class="w-4 h-4" />
          Select File
        </button>
        <input 
          type="file" 
          @change="handleFileChange" 
          accept=".pcap,.log" 
        >
      </div>
      <button 
        @click="uploadFile"
        class="primary w-full"
        :disabled="!selectedFile"
      >
        <Check class="w-4 h-4" />
        Upload
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { Upload, Check } from 'lucide-vue-next'
import axios from 'axios'

const emit = defineEmits(['upload-success'])

const selectedFile = ref(null)

const handleFileChange = (event) => {
  selectedFile.value = event.target.files[0]
}

const uploadFile = async () => {
  if (!selectedFile.value) return

  const formData = new FormData()
  formData.append('file', selectedFile.value)
  try {
    await axios.post('/api/files/upload', formData)
    emit('upload-success')
    selectedFile.value = null
  } catch (error) {
    console.error('File upload failed:', error)
  }
}
</script>

<style scoped>
.file-upload {
  padding: 1.5rem;
}

.file-upload h3 {
  margin-bottom: 1rem;
  font-size: 0.875rem;
  font-weight: 500;
  color: #64748b;
}

.upload-content {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
</style> 