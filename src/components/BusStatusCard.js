import React, { useEffect, useState } from 'react'
import "./BusStatusCard.css"
import { Clock, MapPin } from 'lucide-react'

const BusStatusCard = ({route,time}) => {
    const arrivalTime = time.toDate();
    const busHours = arrivalTime.getHours();
    const busMinutes = arrivalTime.getMinutes().toString().padStart(2, '0');

    const [countdown, setCountdown] = useState('');

    useEffect(() => {
        const calculateCountdown = () => {
            const now = new Date();
            const differenceInSeconds = Math.round((arrivalTime - now) / 1000);

            if (differenceInSeconds < 0) {
                setCountdown("発車済み");
                return;
            }
            //課題点：値が３５８０だったときtimeは０になる
            const hours = Math.floor(differenceInSeconds / 3600)
            const minutes = Math.ceil(differenceInSeconds % 3600 / 60);
            //const seconds = differenceInSeconds % 60;

            //setCountdown(`${minutes}分${seconds.toString().padStart(2, '0')}秒`);
            if(hours > 0){
                if(minutes != 60){
                    setCountdown(`待ち時間：${hours}時間${minutes}分`); 
                }
                else if(minutes == 60){
                    setCountdown(`待ち時間：${hours + 1}時間`); 
                }
            }
            if(hours == 0){
                setCountdown(`待ち時間：${minutes}分`);
            }
        };

        calculateCountdown();

        const intervalId = setInterval(calculateCountdown, 1000);

        return () => clearInterval(intervalId);
    }, [arrivalTime]);

  return (
    <div className='card'>
        <div className='card_info'>
            <div className='route'>
                <MapPin className='icon'/>
                {route}
                <div className="nextTime">
                    <span>{busHours}:{busMinutes}</span>
                </div>
            </div>
            <div className="card_status">
                <Clock className='icon'/>
                <span>
                    {countdown}
                </span> 
            </div>
        </div>
      
    </div>
  )
}

export default BusStatusCard