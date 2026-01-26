import React,{ useEffect, useState } from 'react';
import './App.css';
import Bus_header from './components/Bus_header';
import BusPrediction from './components/BusPrediction';
import BusStatusCard from './components/BusStatusCard';
import BusScheduleTable from './components/BusScheduleTable';

// firebase関連
import db from "./firebase"
import {collection,getDocs, limit, onSnapshot, orderBy, query} from "firebase/firestore"

function App() {

  const[busSchedules,setBusSchedules] = useState([])
  const [displayedSchedules, setDisplayedSchedules] = useState([])
  
  const[people_count,set_people_count] = useState(0) //現在の並んでいる人数

  const[bus_index,set_bus_index] = useState(0) //何本目のバスに乗車できるか

  //データベース
  useEffect(()=>{
    const bus_schedule_data = collection(db,"bus_schedules")
    const q = query(bus_schedule_data,orderBy("stop_time"))

    getDocs(q).then((querySnapshot) =>{
    console.log(querySnapshot.docs.map((doc)=>doc.data()))
    setBusSchedules(querySnapshot.docs.map((doc)=>doc.data()))
    })
  },[])

  //現在の時刻を参照し、4つのデータを代入する
  useEffect(() => {
    if (busSchedules.length === 0) return;

    const now = new Date();

    const createScheduleWithDate = (schedule, date) => {
      const firestoreTimestamp = schedule.stop_time.toDate();
      const newDate = new Date(date);
      newDate.setHours(
        firestoreTimestamp.getHours(),
        firestoreTimestamp.getMinutes(),
        firestoreTimestamp.getSeconds(),
        firestoreTimestamp.getMilliseconds()
      );

      return {
        ...schedule,
        // このオブジェクトがBusPredictionとBusStatusCardに渡される
        stop_time: {//stoptimeを編集
          ...schedule.stop_time,
          toDate: () => newDate,//todateを編集
        },
      };
    };

    const today = new Date();
    const tomorrow = new Date();
    tomorrow.setDate(tomorrow.getDate() + 1);

    const todaySchedules = busSchedules.map(s => createScheduleWithDate(s, today));
    const tomorrowSchedules = busSchedules.map(s => createScheduleWithDate(s, tomorrow));

    const allSchedules = [...todaySchedules, ...tomorrowSchedules];//今日のデータと明日のデータが入った配列

    const upcomingSchedules = allSchedules
      .filter(s => s.stop_time.toDate() > now)
      .sort((a, b) => a.stop_time.toDate() - b.stop_time.toDate());

    setDisplayedSchedules(upcomingSchedules.slice(0, 4));
    console.log("Final displayed schedules for UI:", upcomingSchedules.slice(0, 4).map(s => s.stop_time.toDate()));

  }, [busSchedules]);

  //最新の人数を取得
  useEffect(()=>{
    const countData = collection(db,"people_count")
    const q = query(countData,orderBy("count_time","desc"),limit(1))

    //リアルタイムに更新されない
    getDocs(q).then((querySnapshot)=>{
      const new_people_count = querySnapshot.docs.map((doc)=>doc.data())
      if(new_people_count[0].count === 0){
        set_bus_index(0)
      }
      else if(new_people_count[0].count !== 0){
        set_bus_index(Math.ceil(new_people_count[0].count / 50 -1))//小数点繰上げ
      }
    })

    //リアルタイムに更新される
    onSnapshot(q,(querySnapshot)=>{
      const new_people_count = querySnapshot.docs.map((doc)=>doc.data())
      set_people_count(new_people_count[0].count)
    })
  },[])

  console.log("レンダリング時の最新のdisplayedSchedules:", displayedSchedules);

  return (
    <div className="app">
      {/* タイトル */}
      <Bus_header/>
      {/* 予測所要時間 */}
      <div className="headText">
        予想所要時間
        <span>({bus_index+1}本目に乗車すると仮定)</span>
      </div>
      {displayedSchedules.length > 0 && bus_index < displayedSchedules.length && (
        <BusPrediction 
          people_count={people_count} 
          ride_time={displayedSchedules[bus_index].stop_time} 
          route={displayedSchedules[bus_index].bus_label}
        />
      )}
      {/* 現在のバスの路線別状況 */}
      <div className="headText">
        直近4本のバス情報
      </div>
      <div className='busCards'>
        {displayedSchedules.map((schedules)=>(//繰り返し表示
          <BusStatusCard
          key={schedules.id}
          route={schedules.bus_label}
          time={schedules.stop_time}/>
        ))}
      </div>
      {/* 空き状況の目安 */}

      {/* バスの時刻表 */}
      <div className="headText">
        バス時刻表
      </div>
      <div className='busScheduleTable'>
        <BusScheduleTable busline="直行便"/>
        <BusScheduleTable busline="銀座経由"/>
        <BusScheduleTable busline="県立総合病院経由"/>
        <BusScheduleTable busline="杉江循環"/>
      </div>
    </div>
  );
}

export default App;
