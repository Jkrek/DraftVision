import React from 'react'
import '../../App.css'
import HeroSection from '../HeroSection'
import HowItWorks from '../HowItWorks'
import Cards from '../Cards'
import Explore from '../Explore'

/*
 * Home — composes the Nocturne home screen:
 *   HeroSection  → cinematic hero + live model card + stat band
 *   HowItWorks   → "How the number gets made" + image strip + "Under the hood"
 *   Cards        → "Top of the board" grid + CTA band
 *   Explore      → destination grid leading to every page
 * Navbar / Ticker / Footer are rendered globally by App.js.
 */
function Home() {
  return (
    <>
      <HeroSection />
      <HowItWorks />
      <Explore />
      <Cards />
    </>
  )
}
export default Home
