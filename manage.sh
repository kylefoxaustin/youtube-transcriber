#!/bin/bash
# YouTube Transcriber Management Script

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}"
    echo "╔═══════════════════════════════════════╗"
    echo "║     YouTube Transcriber Manager       ║"
    echo "╚═══════════════════════════════════════╝"
    echo -e "${NC}"
}

print_status() {
    if docker compose ps --quiet 2>/dev/null | grep -q .; then
        echo -e "${GREEN}● Container is running${NC}"
        echo -e "  Web UI: ${BLUE}http://localhost:8000${NC}"
    else
        echo -e "${YELLOW}○ Container is stopped${NC}"
    fi
}

case "${1:-help}" in
    start)
        print_header
        echo "Starting YouTube Transcriber..."
        docker compose up -d
        echo -e "\n${GREEN}✓ Started!${NC}"
        echo -e "Open ${BLUE}http://localhost:8000${NC} in your browser"
        ;;
    
    stop)
        print_header
        echo "Stopping YouTube Transcriber..."
        docker compose down
        echo -e "${GREEN}✓ Stopped${NC}"
        ;;
    
    restart)
        print_header
        echo "Restarting YouTube Transcriber..."
        docker compose restart
        echo -e "${GREEN}✓ Restarted${NC}"
        ;;
    
    logs)
        docker compose logs -f
        ;;
    
    build)
        print_header
        echo "Building Docker image..."
        docker compose build --no-cache
        echo -e "${GREEN}✓ Build complete${NC}"
        ;;
    
    update)
        print_header
        echo "Updating and rebuilding..."
        git pull
        docker compose up -d --build
        echo -e "${GREEN}✓ Updated${NC}"
        ;;
    
    shell)
        docker compose exec transcriber bash
        ;;
    
    gpu)
        print_header
        echo "Checking GPU access..."
        docker compose exec transcriber nvidia-smi
        ;;
    
    status)
        print_header
        print_status
        echo ""
        docker compose ps
        ;;
    
    clean)
        print_header
        echo -e "${YELLOW}This will remove all transcription output!${NC}"
        read -p "Are you sure? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf output/transcripts/*
            rm -rf output/audio/*
            echo -e "${GREEN}✓ Cleaned${NC}"
        fi
        ;;
    
    help|*)
        print_header
        echo "Usage: ./manage.sh <command>"
        echo ""
        echo "Commands:"
        echo "  start     Start the transcriber"
        echo "  stop      Stop the transcriber"
        echo "  restart   Restart the transcriber"
        echo "  logs      View live logs"
        echo "  build     Rebuild the Docker image"
        echo "  update    Pull latest code and rebuild"
        echo "  shell     Open a shell in the container"
        echo "  gpu       Check GPU access"
        echo "  status    Show container status"
        echo "  clean     Remove all transcription output"
        echo "  help      Show this help"
        echo ""
        print_status
        ;;
esac
