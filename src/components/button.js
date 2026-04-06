import React from 'react'
import './Button.css';
import { Link } from 'react-router-dom'

const STYLES = ['btn--primary', 'btn--outline']

const SIZES = ['btn--medium', 'btn--large']


export const Button = ({
    children,
    type = 'button',
    onClick,
    buttonStyle,
    buttonSize,
    to,
    className = '',
    disabled = false
}) => {
    const checkButtonStyle = STYLES.includes(buttonStyle) ? buttonStyle : STYLES[0]
    const checkButtonSize = SIZES.includes(buttonSize) ? buttonSize : SIZES[0]
    const classes = `btn ${checkButtonStyle} ${checkButtonSize} ${className}`.trim()

    if (to) {
        return (
            <Link
                to={to}
                className={`btn-mobile ${classes}`.trim()}
                onClick={disabled ? (event) => event.preventDefault() : onClick}
                aria-disabled={disabled}
            >
                {children}
            </Link>
        )
    }

    return (
        <button
            className={classes}
            onClick={onClick}
            type={type}
            disabled={disabled}
        >
            {children}
        </button>
    )
}
