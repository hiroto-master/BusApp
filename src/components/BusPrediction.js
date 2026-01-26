import React, { useEffect, useState } from 'react'
import "./BusPrediction.css"
import { Clock,MapPin } from 'lucide-react';

const BusPrediction = ({people_count,ride_time,route}) => {
  const arrivalTime = ride_time.toDate();
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

      const hours = Math.floor(differenceInSeconds / 3600)
      const minutes = Math.floor(differenceInSeconds % 3600 / 60);
      const seconds = differenceInSeconds % 60;

      //setCountdown(`${minutes}分${seconds.toString().padStart(2, '0')}秒`);
      if(hours > 0){
        if(minutes != 0){
          setCountdown(`${hours}時間${minutes}分${seconds.toString().padStart(2, '0')}秒後に乗車できます`);
        }
        else if(minutes == 0){
          setCountdown(`${hours}時間${seconds.toString().padStart(2, '0')}秒後に乗車できます`);
        }
      }
      else if(hours == 0){
        if(minutes != 0){
          setCountdown(`${minutes}分${seconds.toString().padStart(2, '0')}秒後に乗車できます`);
        }
        else if(minutes == 0){
          setCountdown(`${seconds.toString().padStart(2, '0')}秒後に乗車できます`);
        }
      }
    };

    calculateCountdown();

    const intervalId = setInterval(calculateCountdown, 1000);

    return () => clearInterval(intervalId);
}, [arrivalTime]);

  return (
    <div className='predictionResult'>
      <div className='backshadow count_people'>
        <div className='countText'>現在並んでいる人数</div>
        <div className='count'>{people_count}</div>
      </div>
      <div className='backshadow prediction_time'>
        <div className='countTime'>
          <Clock size={25}/>
          <div>{countdown}</div>
        </div>
        <div >
          <div className='bus_route'>
            <MapPin size={20}/>
            {route}
            <span>{busHours}:{busMinutes}</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default BusPrediction