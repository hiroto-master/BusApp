import React from 'react'
import "./Bus_header.css"
import {BusFront} from "lucide-react"
import { RefreshCw } from 'lucide-react'
const Bus_header = () => {
  const handleRefresh = () => {
     window.location.reload();
  };
  return (
    <div className='title'>
        <div className='title_text'>
            <BusFront className="icon"/>
            学校バス混雑状況
        </div>
        <div>
            <button onClick={handleRefresh} className="refreshButton">
              <RefreshCw size={16} className="refreshIcon" />
              更新
            </button>
        </div>
    </div>
  )
}

export default Bus_header