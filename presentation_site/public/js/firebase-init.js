/**
 * Firebase init for Fish Finder presentation site (Hosting + optional Analytics).
 * Config is public client-side — restrict domains in Firebase Console if needed.
 */
import { initializeApp } from 'https://www.gstatic.com/firebasejs/11.6.0/firebase-app.js';
import { getAnalytics, isSupported } from 'https://www.gstatic.com/firebasejs/11.6.0/firebase-analytics.js';

const firebaseConfig = {
  apiKey: 'AIzaSyDd9ODk0HqpmqYZDOc__heQQRfrkgncZZo',
  authDomain: 'fishfinder-ad52d.firebaseapp.com',
  projectId: 'fishfinder-ad52d',
  storageBucket: 'fishfinder-ad52d.firebasestorage.app',
  messagingSenderId: '137244928876',
  appId: '1:137244928876:web:cb6c8e1ebe3ccfb4472149',
  measurementId: 'G-P165C27S9B',
};

const app = initializeApp(firebaseConfig);

isSupported().then(function (supported) {
  if (supported) {
    getAnalytics(app);
  }
});
