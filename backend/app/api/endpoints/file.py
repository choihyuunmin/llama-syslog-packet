from fastapi import APIRouter, UploadFile, HTTPException
from typing import Dict, Any
import dpkt
import logging
from datetime import datetime

router = APIRouter()

def analyze_pcap(file_content: bytes) -> Dict[str, Any]:
    """PCAP 파일 분석"""
    try:
        pcap = dpkt.pcap.Reader(file_content)
        packet_count = 0
        protocols = {}
        total_size = 0
        
        for _, buf in pcap:
            packet_count += 1
            total_size += len(buf)
            
            try:
                eth = dpkt.ethernet.Ethernet(buf)
                if isinstance(eth.data, dpkt.ip.IP):
                    protocol = eth.data.p
                    protocols[protocol] = protocols.get(protocol, 0) + 1
            except:
                continue
                
        return {
            "packet_count": packet_count,
            "total_size": total_size,
            "protocols": protocols,
            "file_type": "pcap",
            "analysis_time": datetime.now().isoformat()
        }
    except Exception as e:
        logging.error(f"PCAP 분석 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=400, detail="PCAP 파일 분석 실패")

def analyze_log(file_content: bytes) -> Dict[str, Any]:
    """로그 파일 분석"""
    try:
        content = file_content.decode('utf-8')
        lines = content.split('\n')
        log_count = len(lines)
        
        # syslog 특성 분석
        priorities = {}
        facilities = {}
        for line in lines:
            if line.startswith('<'):
                try:
                    priority = int(line[1:line.find('>')])
                    facility = priority >> 3
                    severity = priority & 0x7
                    
                    priorities[severity] = priorities.get(severity, 0) + 1
                    facilities[facility] = facilities.get(facility, 0) + 1
                except:
                    continue
                    
        return {
            "log_count": log_count,
            "priorities": priorities,
            "facilities": facilities,
            "file_type": "log",
            "analysis_time": datetime.now().isoformat()
        }
    except Exception as e:
        logging.error(f"로그 파일 분석 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=400, detail="로그 파일 분석 실패")

@router.post("/analyze")
async def analyze_file(file: UploadFile):
    """파일 업로드 및 분석"""
    try:
        content = await file.read()
        file_extension = file.filename.split('.')[-1].lower()
        
        if file_extension == 'pcap':
            return analyze_pcap(content)
        elif file_extension in ['log', 'txt']:
            return analyze_log(content)
        else:
            raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식입니다.")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 