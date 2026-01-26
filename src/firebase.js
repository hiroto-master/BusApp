import {initializeApp} from "firebase/app"
import {getFirestore} from "firebase/firestore"
import { getAnalytics } from "firebase/analytics";

const firebaseConfig = {

    // firebaseのnpmをここに貼り付ける
    
}

const app = initializeApp(firebaseConfig)
const db = getFirestore(app)
const analytics = getAnalytics(app);

export default db;