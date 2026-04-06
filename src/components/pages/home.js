import React from 'react'
import '../../App.css'
import HeroSection from '../HeroSection'
import HowItWorks from '../HowItWorks'
import Cards from '../Cards'
import Footer from '../Footer'

function Home() {
  return (
    <>
      <HeroSection />
      <HowItWorks />
      <Cards />
      <Footer />
    </>
  )
}
export default Home
