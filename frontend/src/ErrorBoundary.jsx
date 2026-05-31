import React from "react";


class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(error, info) {
    console.error("Frontend render error", error, info);
  }

  handleReload = () => {
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="shell auth-shell">
          <div className="auth-card">
            <h2>Noe gikk galt</h2>
            <p className="muted">
              Appen traff en uventet feil under visning. Last siden på nytt og prøv igjen.
            </p>
            <button className="primary-button" onClick={this.handleReload}>
              Last siden på nytt
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}


export default ErrorBoundary;
