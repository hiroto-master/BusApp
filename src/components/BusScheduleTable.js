import React, { useState, useEffect } from 'react';
import './BusScheduleTable.css';
import db from '../firebase'; // firebase.jsをインポート
import { collection, getDocs, query, orderBy, where } from 'firebase/firestore';

const BusScheduleTable = ({ busline }) => {
  const [scheduleData, setScheduleData] = useState([]);

  useEffect(() => {
    if (!busline) return; // buslineが渡されていない場合は何もしない

    const fetchSchedule = async () => {
      try {
        const q = query(
          collection(db, "bus_schedules"),
          where("bus_label", "==", busline), // {busline} を busline に修正
          orderBy("stop_time")
        );
        const querySnapshot = await getDocs(q);

        const data = querySnapshot.docs.map(doc => doc.data());

        const groupedByHour = {};
        data.forEach(item => {
          if (item.stop_time) {
            const date = item.stop_time.toDate();
            const hour = date.getHours();
            const minute = date.getMinutes().toString().padStart(2, '0');

            if (!groupedByHour[hour]) {
              groupedByHour[hour] = [];
            }
            groupedByHour[hour].push(minute);
          }
        });

        const formattedSchedule = Object.keys(groupedByHour)
          .sort((a, b) => a - b)
          .map(hour => ({
            hour: parseInt(hour),
            minutes: groupedByHour[hour].sort((a, b) => a - b).join(' '),
          }));

        setScheduleData(formattedSchedule);
      } catch (error) {
        console.error("Error fetching schedule: ", error);
      }
    };

    fetchSchedule();
  }, [busline]); // 依存配列にbuslineを追加

  return (
    <div className='table-container'>
      <div className='table-title'>{busline}</div>
      <div className='schedule-wrapper'>
        <div className='schedule'>
          <table className='schedule-table'>
            <thead>
              <tr>
                <th className='hour'>時</th>
                <th className='minute'>分</th>
              </tr>
            </thead>
            <tbody>
              {scheduleData.map((item, index) => (
                <tr key={index}>
                  <td className='hour'>{item.hour}</td>
                  <td className='minute'>{item.minutes}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default BusScheduleTable;