# Copilot Instructions - Parking Management System

## Architecture Overview
This is a Flask-based parking management system with AI vision capabilities and payment integration:
- **Flask app** with Blueprint-based routing (`routes/`) 
- **AI services** for license plate OCR and parking slot detection (`services/vision_service.py`, `utils/ocr.py`)
- **Database layer** using raw PyMySQL with manual schema management (`DB/connection.py`)
- **Payment integration** with TossPayments API
- **Flutter mobile app** in `parking_empty_mlkit/`

## Key Patterns & Conventions

### Database Access Pattern
- Use `get_db()` from `DB/connection.py` for raw PyMySQL connections with dict cursors
- Manual transaction control with `autocommit=False` - always commit/rollback explicitly
- Schema files in `DB/` folder (`.sql` files) are loaded automatically on first connection
- **No ORM** - write raw SQL queries, not SQLAlchemy ORM calls

### Service Layer Architecture
- **OCR Pipeline**: `utils/ocr.py` → `utils/OCR_engines/` → Google Vision API
- **Vision AI**: `services/vision_service.py` uses PyTorch + timm models loaded globally at startup
- **Parking Logic**: `services/parking_service.py` handles entry/exit with dependency injection for testing
- **Fee Calculation**: `utils/fee_calc.py` with user type-based pricing (non_member, member_regular, member_subscriber)

### Error Handling & Responses
- Use `utils/response.py` functions: `make_response()` and `error_response()`
- Blueprint-level error handlers in route files (see `parking_routes.py`)
- OCR failures should raise `ValueError` with descriptive messages

### Testing Strategy
- Unit tests in `test/` folder focus on business logic (fee calculation, OCR)
- Use dependency injection in services for mocking (e.g., `ocr_func` parameter in `handle_entry()`)
- Test files follow `test_*.py` naming convention

### AI Model Integration
- Models loaded once at module import in `services/vision_service.py`
- Global variables for model, device, and transforms to avoid reload overhead  
- Always call `model.eval()` for inference mode
- Image preprocessing must match training pipeline exactly

### Configuration Management
- Database config in `config.py` (hardcoded for dev)
- Environment variables for API keys (`.env` file)
- AI model settings (MODEL_NAME, NUM_CLASSES) in `config.py`

## Development Workflows

### Running the Application
```bash
python app.py  # Main Flask server
python server.py  # Alternative entry point with additional setup
```

### Database Setup
- Database/tables created automatically on first `get_db()` call
- Apply schema changes by modifying `.sql` files in `DB/` folder
- Use `scripts/apply_notices_sql.py` for data migrations

### Adding New Routes
1. Create Blueprint in `routes/` folder
2. Register in `app.py` with `app.register_blueprint()`
3. Follow existing error handling patterns
4. Use service layer for business logic

### AI Model Updates
- Replace model file in `ml_models/` directory
- Update MODEL_NAME in `config.py` if architecture changes
- Restart server to reload global model variables

## Key Files to Understand
- `services/parking_service.py` - Core parking entry/exit logic
- `utils/fee_calc.py` - Pricing rules and calculations  
- `DB/connection.py` - Database connection and schema management
- `services/vision_service.py` - AI model loading and inference
- `routes/parking_routes.py` - API endpoints and request handling